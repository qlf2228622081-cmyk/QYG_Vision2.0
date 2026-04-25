#include "buff_detector.hpp"

#include "tools/logger.hpp"

namespace auto_buff
{

/**
 * @brief Buff_Detector 构造函数
 */
Buff_Detector::Buff_Detector(const std::string & config) : status_(LOSE), lose_(0), MODE_(config) {}

/**
 * @brief 传统视觉预处理逻辑
 * 逻辑：灰度化 -> 粗二值化 -> 膨胀，目的是使 R 标和扇叶边缘更加连贯，便于寻找闭合轮廓。
 */
void Buff_Detector::handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img)
{
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, 100, 255, cv::THRESH_BINARY);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), 1);
}

/**
 * @brief 计算并修正能量机关的旋转中心 (R 标)
 * 逻辑：
 * 1. 利用神经网络输出的扇叶角点 5 (中心) 和 6 (方向参考点) 初步推算出 R 标的大致位置。
 * 2. 构造一个圆形 Mask，聚焦于推算位置周围。
 * 3. 在 Mask 范围内寻找轮廓，通过“形状比例”和“距离推测点偏差”综合权重，筛选出真正的 R 标中心坐标。
 */
cv::Point2f Buff_Detector::get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img)
{
  if (fanblades.empty()) {
    tools::logger()->debug("[Buff_Detector] 无法计算r_center!");
    return {0, 0};
  }

  // 1. 初步估计 R 标大致位置
  cv::Point2f r_center_estimate = {0, 0};
  for (auto & fanblade : fanblades) {
    auto point5 = fanblade.points[4]; // 扇叶中心
    auto point6 = fanblade.points[5]; // 方向点
    // 基于几何比例 (1.4) 线外延推算 R 标
    r_center_estimate += (point6 - point5) * 1.4 + point5;
  }
  r_center_estimate /= float(fanblades.size());

  // 2. 传统视觉精细化定位
  cv::Mat dilated_img;
  handle_img(bgr_img, dilated_img);
  
  // 构造局部搜索遮罩 (Mask)
  double radius = cv::norm(fanblades[0].points[2] - fanblades[0].center) * 0.8;
  cv::Mat mask = cv::Mat::zeros(dilated_img.size(), CV_8U);
  circle(mask, r_center_estimate, radius, cv::Scalar(255), -1);
  bitwise_and(dilated_img, mask, dilated_img);

  // 在二值图中查找最符合 R 标特征的轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::Point2f r_center_final = r_center_estimate;
  cv::findContours(dilated_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  
  double min_error_weight = INF;
  for (auto & it : contours) {
    auto rotated_rect = cv::minAreaRect(it);
    // R 标应近似为正方形 (比例接近 1:1)
    double aspect_ratio = rotated_rect.size.height > rotated_rect.size.width
                            ? rotated_rect.size.height / rotated_rect.size.width
                            : rotated_rect.size.width / rotated_rect.size.height;
    
    // 综合评价分数：比例越接近 1 且 离估算位置越近，分数越低（越优）
    double score = aspect_ratio + cv::norm(rotated_rect.center - r_center_estimate) / (radius / 3);
    if (score < min_error_weight) {
      min_error_weight = score;
      r_center_final = rotated_rect.center;
    }
  }
  return r_center_final;
};

/**
 * @brief 目标丢失状态机管理
 */
void Buff_Detector::handle_lose()
{
  lose_++;
  if (lose_ >= LOSE_MAX) {
    status_ = LOSE;
    last_powerrune_ = std::nullopt; // 重置一切历史状态
  } else {
    status_ = TEM_LOSE; // 处于“临时丢失”宽限期内
  }
}

/**
 * @brief 多目标候选模式 (主要接口)
 * 解析流程：DL 推理 -> 属性转换 -> 计算 R 标 -> 时序匹配生成 PowerRune。
 */
std::optional<PowerRune> Buff_Detector::detect_24(cv::Mat & bgr_img)
{
  // 1. 获取所有亮起的扇叶候选框
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  // 2. 转换数据结构
  std::vector<FanBlade> fanblades;
  for (auto & result : results) 
    fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));

  // 3. 计算旋转中心并构建 PowerRune 对象
  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);

  // 4. 异常处理：几何逻辑冲突（如无法识别最新 Target）
  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  // 5. 更新状态信息
  status_ = TRACK;
  lose_ = 0;
  last_powerrune_ = std::make_optional<PowerRune>(powerrune);
  return last_powerrune_;
}

/**
 * @brief 单目标模式 (主要用于快速回归)
 */
std::optional<PowerRune> Buff_Detector::detect(cv::Mat & bgr_img)
{
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_onecandidatebox(bgr_img);

  if (results.empty()) {
    handle_lose();
    return std::nullopt;
  }

  std::vector<FanBlade> fanblades;
  auto result = results[0];
  fanblades.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));

  auto r_center = get_r_center(fanblades, bgr_img);
  PowerRune powerrune(fanblades, r_center, last_powerrune_);

  if (powerrune.is_unsolve()) {
    handle_lose();
    return std::nullopt;
  }

  status_ = TRACK;
  lose_ = 0;
  last_powerrune_ = std::make_optional<PowerRune>(powerrune);
  return last_powerrune_;
}

/**
 * @brief 调试接口：带物理约束的识别
 */
std::optional<PowerRune> Buff_Detector::detect_debug(cv::Mat & bgr_img, cv::Point2f v)
{
  std::vector<YOLO11_BUFF::Object> results = MODE_.get_multicandidateboxes(bgr_img);
  if (results.empty()) return std::nullopt;

  std::vector<FanBlade> fanblades_t;
  for (auto & result : results)
    fanblades_t.emplace_back(FanBlade(result.kpt, result.kpt[4], _light));

  auto r_center = get_r_center(fanblades_t, bgr_img);
  
  std::vector<FanBlade> fanblades;
  for (auto & fanblade : fanblades_t) {
    // 筛选出离指定方向向量 v 最近的扇叶
    if (cv::norm((fanblade.center - r_center) - v) < 10 || results.size() == 1) {
      fanblades.emplace_back(fanblade);
      break;
    }
  }
  
  if (fanblades.empty()) return std::nullopt;
  
  PowerRune powerrune(fanblades, r_center, std::nullopt);
  return std::make_optional<PowerRune>(powerrune);
}

}  // namespace auto_buff