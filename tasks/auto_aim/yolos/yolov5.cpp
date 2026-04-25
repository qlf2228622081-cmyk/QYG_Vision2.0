#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

/**
 * @brief YOLOv5 构造函数
 * 使用 OpenVINO 的 PrePostProcessor 构建高效的推理流水线。
 */
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  // 1. 加载参数
  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  
  // 加载静态 ROI 配置 (如果 use_roi 为 true)
  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // 2. OpenVINO 模型初始化与预处理管道构建
  auto model = core_.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 配置输入数据格式：NHWC (BGR) -> 模型需要 NCHW (RGB/Normalized)
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  // 定义预处理算子 (直接由 OpenVINO 在硬件加速器上执行)
  input.preprocess()
    .convert_element_type(ov::element::f32) // 转为浮点
    .convert_color(ov::preprocess::ColorFormat::RGB) // BGR 转 RGB
    .scale(255.0); // 归一化

  model = ppp.build();
  // 编译模型提升推理效率 (LATENCY 模式)
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

/**
 * @brief 执行推理检测流程
 */
std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  // 1. ROI 裁切逻辑
  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // 2. 图像缩放处理 (保持长宽比缩放到 640x640)
  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

  // 3. 执行硬件推理
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 4. 获取推理结果并解析
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

/**
 * @brief 解析模型输出张量 (Post-processing)
 * YOLOv5-Armor 输出通常包含：xywh, confidence, color_scores, class_scores, 4-keypoints。
 */
std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  // 1. 遍历每一行检测到的 Anchor
  for (int r = 0; r < output.rows; r++) {
    double score = output.at<float>(r, 8); // 获取置信度得分
    score = sigmoid(score);

    if (score < score_threshold_) continue;

    // 2. 提取类别和颜色概率
    cv::Mat color_scores = output.row(r).colRange(9, 13);
    cv::Mat classes_scores = output.row(r).colRange(13, 22);
    cv::Point class_id, color_id;
    double score_num, score_color;
    cv::minMaxLoc(classes_scores, NULL, &score_num, NULL, &class_id);
    cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);

    // 3. 提取 4 个角点的回归坐标并还原比例
    std::vector<cv::Point2f> armor_key_points;
    armor_key_points.push_back(cv::Point2f(output.at<float>(r, 0) / scale, output.at<float>(r, 1) / scale));
    armor_key_points.push_back(cv::Point2f(output.at<float>(r, 6) / scale, output.at<float>(r, 7) / scale));
    armor_key_points.push_back(cv::Point2f(output.at<float>(r, 4) / scale, output.at<float>(r, 5) / scale));
    armor_key_points.push_back(cv::Point2f(output.at<float>(r, 2) / scale, output.at<float>(r, 3) / scale));

    // 计算外接矩形
    float min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10;
    for (const auto & pt : armor_key_points) {
      min_x = std::min(min_x, pt.x); max_x = std::max(max_x, pt.x);
      min_y = std::min(min_y, pt.y); max_y = std::max(max_y, pt.y);
    }
    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(color_id.x);
    num_ids.emplace_back(class_id.x);
    boxes.emplace_back(rect);
    confidences.emplace_back(score);
    armors_key_points.emplace_back(armor_key_points);
  }

  // 4. 执行非极大值抑制 (NMS)
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // 5. 对最终结果进行合法性过滤和角点细化
  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    // [关键步骤] 如果开启传统校准，则使用几何轮廓进一步微调角点坐标
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

/**
 * @brief 校验识别名称是否合法
 */
bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

/**
 * @brief 校验尺寸类型与兵种是否匹配（如：英雄一定是大大，步兵可以是大大或小小）
 */
bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);
  return name_ok;
}

/**
 * @brief 坐标归一化
 */
cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

/**
 * @brief 结果绘制 (调试用)
 */
void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::rectangle(detection, roi_, {0, 255, 0}, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

/**
 * @brief 保存异常图像
 */
void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

/**
 * @brief Sigmoid 激活函数处理输出得分
 */
double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

/**
 * @brief 外部统一调用接口
 */
std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim