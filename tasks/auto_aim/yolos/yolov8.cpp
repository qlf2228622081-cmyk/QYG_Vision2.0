#include "yolov8.hpp"

#include <fmt/chrono.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "tasks/auto_aim/classifier.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

/**
 * @brief YOLOv8 构造函数
 * 构建基于 OpenVINO 的目标检测管道，输入分辨率通常配置为 416x416。
 */
YOLOV8::YOLOV8(const std::string & config_path, bool debug)
: classifier_(config_path), detector_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  // 1. 加载参数
  model_path_ = yaml["yolov8_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  
  // 静态 ROI 配置
  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  // 2. OpenVINO 预处理流水线
  auto model = core_.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 416, 416, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

/**
 * @brief 执行推理检测流程
 */
std::list<Armor> YOLOV8::detect(const cv::Mat & raw_img, int frame_count)
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

  // 2. 图像缩放处理 (416x416)
  auto x_scale = static_cast<double>(416) / bgr_img.rows;
  auto y_scale = static_cast<double>(416) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  auto input = cv::Mat(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});

  // 3. 硬件推理
  ov::Tensor input_tensor(ov::element::u8, {1, 416, 416, 3}, input.data);
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 4. 获取输出结果
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

/**
 * @brief 解析 YOLOv8 输出张量 (Post-processing)
 * YOLOv8 模型输出结构与 v5 有所不同，通常第一步需要执行转置 (Transpose)。
 */
std::list<Armor> YOLOV8::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // YOLOv8 输出通常为 [batch, features, anchors], 需要转置为 [anchors, features]
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  // 1. 遍历每个 Anchor 的输出特征
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 14);

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    if (score < score_threshold_) continue;

    // 坐标还原
    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    // 关键点提取
    std::vector<cv::Point2f> armor_key_points;
    for (int i = 0; i < 4; i++) {
        armor_key_points.push_back({one_key_points.at<float>(0, i * 2 + 0) / (float)scale, 
                                     one_key_points.at<float>(0, i * 2 + 1) / (float)scale});
    }

    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  // 2. 非极大值抑制 (NMS)
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  // 3. 构建 Armor 对象并执行时序一致性处理
  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]); // [重要] 对角点进行固定排序
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // 4. [分类器集成] 利用独立的 Classifier 识别兵种数字
  for (auto it = armors.begin(); it != armors.end();) {
    it->pattern = get_pattern(bgr_img, *it); // 提取中心图案
    classifier_.classify(*it); // CNN 识别

    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->type = get_type(*it); // 推断大/小装甲板
    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

/**
 * @brief 校验名称及置信度
 */
bool YOLOV8::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;
  return name_ok && confidence_ok;
}

/**
 * @brief 校验版本/兵种匹配性
 */
bool YOLOV8::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);
  return name_ok;
}

/**
 * @brief 基于兵种逻辑判断装甲板类型
 */
ArmorType YOLOV8::get_type(const Armor & armor)
{
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) return ArmorType::big;
  
  if (armor.name == ArmorName::two || armor.name == ArmorName::sentry ||
      armor.name == ArmorName::outpost) return ArmorType::small;

  return ArmorType::small; // 默认步兵假设为小
}

/**
 * @brief 归一化中心坐标轴
 */
cv::Point2f YOLOV8::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / (float)bgr_img.cols, center.y / (float)bgr_img.rows};
}

/**
 * @brief 提取装甲板中心图案 (Pattern)
 * 逻辑：基于回归的角点，计算出能够包围数字区域的 ROI 窗口。
 */
cv::Mat YOLOV8::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  // 利用灯条比例 (1.125) 向上下延伸，确保数字完整出现在 Pattern 中
  auto tl = (armor.points[0] + armor.points[3]) / 2 - (armor.points[3] - armor.points[0]) * 1.125;
  auto bl = (armor.points[0] + armor.points[3]) / 2 + (armor.points[3] - armor.points[0]) * 1.125;
  auto tr = (armor.points[2] + armor.points[1]) / 2 - (armor.points[2] - armor.points[1]) * 1.125;
  auto br = (armor.points[2] + armor.points[1]) / 2 + (armor.points[2] - armor.points[1]) * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  
  auto roi = cv::Rect(cv::Point(roi_left, roi_top), cv::Point(roi_right, roi_bottom));

  if (roi.width <= 0 || roi.height <= 0 || roi.x < 0 || roi.y < 0 ||
      roi.x + roi.width > bgr_img.cols || roi.y + roi.height > bgr_img.rows) return cv::Mat();

  return bgr_img(roi).clone();
}

/**
 * @brief 保存异常样本
 */
void YOLOV8::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, armor.pattern);
}

/**
 * @brief 绘制检测结果
 */
void YOLOV8::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) cv::rectangle(detection, roi_, {0, 255, 0}, 2);
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

/**
 * @brief 对回归的关键点进行顺时针排序 (TL, TR, BR, BL)
 */
void YOLOV8::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) return;

  // 1. 先按 Y 排序区分上下
  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  // 2. 再按 X 排序区分左右
  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(bottom_points.begin(), bottom_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  // 3. 填回固定顺序
  keypoints[0] = top_points[0];     // TL
  keypoints[1] = top_points[1];     // TR
  keypoints[2] = bottom_points[1];  // BR
  keypoints[3] = bottom_points[0];  // BL
}

/**
 * @brief 外部统一接口
 */
std::list<Armor> YOLOV8::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim