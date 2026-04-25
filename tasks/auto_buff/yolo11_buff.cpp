#include "yolo11_buff.hpp"

// 检测阈值配置
const double ConfidenceThreshold = 0.7f;
const double IouThreshold = 0.4f;

namespace auto_buff
{

/**
 * @brief YOLO11_BUFF 构造函数
 * 初始化 OpenVINO 推理引擎并准备输入张量。
 */
YOLO11_BUFF::YOLO11_BUFF(const std::string & config)
{
  auto yaml = YAML::LoadFile(config);
  std::string model_path = yaml["model"].as<std::string>();
  
  // 加载并编译模型 (默认为 CPU，若支持也可配置位 GPU/MYRIAD/NPU)
  model = core.read_model(model_path);
  compiled_model = core.compile_model(model, "CPU");
  
  infer_request = compiled_model.create_infer_request();
  input_tensor = infer_request.get_input_tensor();
  
  // 统一输入尺寸为 640x640
  input_tensor.set_shape({1, 3, 640, 640});
}

/**
 * @brief 多目标候选框检测接口 (核心流程)
 */
std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  const int64 start = cv::getTickCount();

  if (image.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::vector<YOLO11_BUFF::Object>();
  }

  // 1. 图像预处理与尺寸调整 (计算缩放因子以保持比例)
  cv::Mat bgr_img = image;
  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  double factor = 1.0 / scale; // 用于将检测坐标映射回原图

  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  
  // 2. 拷贝数据到 OpenVINO Tensor
  ov::Tensor p_input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);
  // 注意：此处若使用 ov::preprocess 可以优化拷贝开销，目前为硬拷贝
  
  // 3. 执行推理
  infer_request.infer();

  // 4. 解析输出张量 (Shape: [15, 8400] -> 4角点坐标 + 1置信度 + NUM_POINTS*2关键点)
  const ov::Tensor output = infer_request.get_output_tensor();
  const ov::Shape output_shape = output.get_shape();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output_shape[1];
  const int out_cols = output_shape[2];
  
  const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);
  
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<float>> objects_keypoints;

  // 遍历所有生成的预测框（Anchors）
  for (int i = 0; i < det_output.cols; ++i) {
    const float score = det_output.at<float>(4, i);
    if (score > ConfidenceThreshold) {
      // 提取核心坐标并映射回原图
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      
      cv::Rect box;
      box.x = static_cast<int>((cx - 0.5 * ow) * factor);
      box.y = static_cast<int>((cy - 0.5 * oh) * factor);
      box.width = static_cast<int>(ow * factor);
      box.height = static_cast<int>(oh * factor);
      boxes.push_back(box);
      confidences.push_back(score);

      // 提取回归的关键点
      std::vector<float> keypoints;
      cv::Mat kpts = det_output.col(i).rowRange(5, 5 + NUM_POINTS * 2);
      for (int j = 0; j < NUM_POINTS; ++j) {
        keypoints.push_back(kpts.at<float>(j * 2 + 0, 0) * factor);
        keypoints.push_back(kpts.at<float>(j * 2 + 1, 0) * factor);
      }
      objects_keypoints.push_back(keypoints);
    }
  }

  // 5. NMS 非极大值抑制处理重叠
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, ConfidenceThreshold, IouThreshold, indexes);

  // 6. 构造最终对象列表并进行可视化绘制
  std::vector<Object> object_result;
  for (size_t i = 0; i < indexes.size(); ++i) {
    const int index = indexes[i];
    Object obj;
    obj.rect = boxes[index];
    obj.prob = confidences[index];

    const std::vector<float> & keypoint = objects_keypoints[index];
    for (int j = 0; j < NUM_POINTS; ++j) {
      obj.kpt.push_back(cv::Point2f(keypoint[j * 2], keypoint[j * 2 + 1]));
    }
    object_result.push_back(obj);

    // 绘制 UI 信息
    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1);
    const std::string label = "buff:" + std::to_string(obj.prob).substr(0, 4);
    cv::putText(image, label, obj.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
    
    for (int j = 0; j < NUM_POINTS; ++j) {
      cv::circle(image, obj.kpt[j], 2, cv::Scalar(255, 0, 0), -1);
    }
  }

  // 计算实时推理性帧率 (FPS)
  const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(image, cv::format("FPS: %.2f", 1.0 / t), {20, 40}, cv::FONT_HERSHEY_PLAIN, 2.0, {255, 0, 0}, 2);

  return object_result;
}

/**
 * @brief 置信度优先单目标检测接口
 */
std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  const int64 start = cv::getTickCount();
  const float factor = fill_tensor_data_image(input_tensor, image);

  infer_request.infer();

  const ov::Tensor output = infer_request.get_output_tensor();
  const float * output_buffer = output.data<const float>();
  const int out_rows = output.get_shape()[1];
  const int out_cols = output.get_shape()[2];
  const cv::Mat det_output(out_rows, out_cols, CV_32F, (float *)output_buffer);

  // 1. 寻找全局置信度最高的目标索引
  int best_index = -1;
  float max_confidence = 0.0f;
  for (int i = 0; i < det_output.cols; ++i) {
    const float confidence = det_output.at<float>(4, i);
    if (confidence > max_confidence) {
      max_confidence = confidence;
      best_index = i;
    }
  }

  // 2. 解析该最优目标
  std::vector<Object> object_result;
  if (max_confidence > ConfidenceThreshold) {
    Object obj;
    const float cx = det_output.at<float>(0, best_index);
    const float cy = det_output.at<float>(1, best_index);
    const float ow = det_output.at<float>(2, best_index);
    const float oh = det_output.at<float>(3, best_index);
    obj.rect = cv::Rect(static_cast<int>((cx - 0.5 * ow) * factor), static_cast<int>((cy - 0.5 * oh) * factor), 
                        static_cast<int>(ow * factor), static_cast<int>(oh * factor));
    obj.prob = max_confidence;
    
    cv::Mat kpts = det_output.col(best_index).rowRange(5, 5 + NUM_POINTS * 2);
    for (int i = 0; i < NUM_POINTS; ++i) {
      obj.kpt.push_back({kpts.at<float>(i * 2, 0) * factor, kpts.at<float>(i * 2 + 1, 0) * factor});
    }
    object_result.push_back(obj);

    // 辅助分析：保存低置信度图片
    if (max_confidence < 0.7) save(std::to_string(start), image);
  }

  return object_result;
}

/**
 * @brief 手边图片转换工具 (BGR -> RGB, 归一化)
 */
void YOLO11_BUFF::convert(
  const cv::Mat & input, cv::Mat & output, const bool normalize, const bool BGR2RGB) const
{
  input.convertTo(output, CV_32F);
  if (normalize) output = output / 255.0;
  if (BGR2RGB) cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
}

/**
 * @brief 填充 OpenVINO 推理张量 (手动布局转换与 Letterbox)
 * 注意：OpenVINO 也可以通过 PrePostProcessor 硬件实现布局转换，目前代码为 CPU 循环实现。
 */
float YOLO11_BUFF::fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const
{
  const ov::Shape tensor_shape = input_tensor.get_shape();
  const size_t num_channels = tensor_shape[1];
  const size_t height = tensor_shape[2];
  const size_t width = tensor_shape[3];
  
  const float scale = std::min(height / float(input_image.rows), width / float(input_image.cols));
  const cv::Matx23f matrix{scale, 0.0, 0.0, 0.0, scale, 0.0};
  
  cv::Mat blob_image;
  cv::warpAffine(input_image, blob_image, matrix, cv::Size(width, height));
  convert(blob_image, blob_image, true, true);

  float * const input_tensor_data = input_tensor.data<float>();
  // 转换 HWC 为 CHW 并填入张量
  for (size_t c = 0; c < num_channels; c++) {
    for (size_t h = 0; h < height; h++) {
      for (size_t w = 0; w < width; w++) {
        input_tensor_data[c * width * height + h * width + w] =
          blob_image.at<cv::Vec<float, 3>>(h, w)[c];
      }
    }
  }
  return 1.0f / scale;
}

// ... 省略 printInputAndOutputsInfo 和 save 辅助调试函数的具体注释，其逻辑相对独立且通用 ...
}  // namespace auto_buff