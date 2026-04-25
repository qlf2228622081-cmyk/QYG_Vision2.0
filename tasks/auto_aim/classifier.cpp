#include "classifier.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{

/**
 * @brief Classifier 构造函数
 * 从参数配置文件加载 ONNX 模型，并初始化两个推理后端。
 */
Classifier::Classifier(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto model = yaml["classify_model"].as<std::string>();

  // 1. 初始化 OpenCV DNN 后端
  net_ = cv::dnn::readNetFromONNX(model);
  
  // 2. 初始化 OpenVINO 后端（配置为优先追求低延迟）
  auto ovmodel = core_.read_model(model);
  compiled_model_ = core_.compile_model(
    ovmodel, "AUTO", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

/**
 * @brief 使用 OpenCV DNN 进行数字识别
 * 流程：灰度化 -> 保持比例缩放至 32x32 -> 归一化 -> 推理 -> Softmax 输出。
 */
void Classifier::classify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  // 1. 预处理：灰度化
  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  // 2. 图像对齐与缩放 (保持比例在 32x32 画布内)
  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});

  // 3. 构建推理 Blob (归一化到 [0, 1])
  auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

  // 4. 网络推理
  net_.setInput(blob);
  cv::Mat outputs = net_.forward();

  // 5. 后处理：Softmax 转概率值
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  // 6. 查找最大概率对应的标签 (Point.x)
  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  // 7. 更新装甲板属性
  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

/**
 * @brief 使用 OpenVINO 进行数字识别 (高性能版)
 */
void Classifier::ovclassify(Armor & armor)
{
  if (armor.pattern.empty()) {
    armor.name = ArmorName::not_armor;
    return;
  }

  // 1. 预处理 (同上)
  cv::Mat gray;
  cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

  auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
  auto x_scale = static_cast<double>(32) / gray.cols;
  auto y_scale = static_cast<double>(32) / gray.rows;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(gray.rows * scale);
  auto w = static_cast<int>(gray.cols * scale);

  if (h == 0 || w == 0) {
    armor.name = ArmorName::not_armor;
    return;
  }

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(gray, input(roi), {w, h});
  input.convertTo(input, CV_32F, 1.0 / 255.0);

  // 2. 构建 OpenVINO Tensor
  ov::Tensor input_tensor(ov::element::f32, {1, 1, 32, 32}, input.data);

  // 3. 异步推理（此处为阻塞调用）
  ov::InferRequest infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 4. 获取输出
  auto output_tensor = infer_request.get_output_tensor();
  cv::Mat outputs(1, 9, CV_32F, output_tensor.data());

  // 5. Softmax 概率化
  float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
  cv::exp(outputs - max, outputs);
  float sum = cv::sum(outputs)[0];
  outputs /= sum;

  // 6. 解析标签
  double confidence;
  cv::Point label_point;
  cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
  int label_id = label_point.x;

  armor.confidence = confidence;
  armor.name = static_cast<ArmorName>(label_id);
}

}  // namespace auto_aim