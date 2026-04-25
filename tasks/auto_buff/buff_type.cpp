#include "buff_type.cpp"

#include <algorithm>

#include "tools/logger.hpp"
namespace auto_buff
{

/**
 * @brief FanBlade 构造函数
 */
FanBlade::FanBlade(
  const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t)
: center(keypoints_center), type(t)
{
  // 插入回归出的角点坐标
  points.insert(points.end(), kpt.begin(), kpt.end());
}

/**
 * @brief 创建空或特定类型的占位扇叶
 */
FanBlade::FanBlade(FanBlade_type t) : type(t)
{
  if (t != _unlight) exit(-1); // 仅允许 _unlight 类型的占位符
}

/**
 * @brief PowerRune 构造函数
 * 逻辑核心：从当前识别到的所有亮起扇叶中找出哪一个是真正的“待击打目标（Target）”。
 */
PowerRune::PowerRune(
  std::vector<FanBlade> & ts, const cv::Point2f center, std::optional<PowerRune> last_powerrune)
: r_center(center), light_num(ts.size())
{
  /// 1. 识别 Target 扇叶分类逻辑
  
  // A. 只有一片亮起，直接认定为 Target
  if (light_num == 1) ts[0].type = _target;
  
  // B. 亮起的扇叶数量没变：找离上一帧 Target 最近的那片（锁定已有目标）
  else if (last_powerrune.has_value() && ts.size() == last_powerrune.value().light_num) {
    auto last_target_center = last_powerrune.value().fanblades[0].center;
    auto target_fanblade_it = ts.begin();
    float min_distance = norm(ts[0].center - last_target_center);
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float distance = norm(it->center - last_target_center);
      if (distance < min_distance) {
        min_distance = distance;
        target_fanblade_it = it;
      }
    }
    target_fanblade_it->type = _target;
    std::iter_swap(ts.begin(), target_fanblade_it); // 将 Target 交换到数组首位
  }
  
  // C. 亮起的扇叶数量增加了：找离所有旧扇叶都最远的那片（识别新亮起的激活目标）
  else if (last_powerrune.has_value() && light_num == (int)last_powerrune.value().light_num + 1) {
    auto last_fanblades = last_powerrune.value().fanblades;
    float max_min_distance = -1.0f;
    auto target_fanblade_it = ts.begin();
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      float min_distance = std::numeric_limits<float>::max();
      for (const auto & last_fanblade : last_fanblades) {
        if (last_fanblade.type == _unlight) continue;
        float distance = norm(it->center - last_fanblade.center);
        if (distance < min_distance) min_distance = distance;
      }
      // 新亮起的叶片中心距离已有叶片中心的投影距离通常是最大的
      if (min_distance > max_min_distance) {
        max_min_distance = min_distance;
        target_fanblade_it = it;
      }
    }
    target_fanblade_it->type = _target;
    std::iter_swap(ts.begin(), target_fanblade_it);
  }
  else {
    // 逻辑异常（如跳帧严重或检测结果冲突）
    tools::logger()->debug("[PowerRune] 识别出错!");
    unsolvable_ = true;
    return;
  }

  /// 2. 计算扇叶相对于 R 标的偏角 (Polar Angle)
  double angle = atan_angle(ts[0].center);
  for (auto & t : ts) {
    t.angle = atan_angle(t.center) - angle;
    if (t.angle < -1e-3) t.angle += CV_2PI;
  }

  /// 3. 合理化扇叶排列顺序 (顺时针 5 分位排列)
  std::sort(ts.begin(), ts.end(), [](const FanBlade & a, const FanBlade & b) {
    return a.angle < b.angle;
  });

  const std::vector<double> target_angles = {
    0, 2.0 * CV_PI / 5.0, 4.0 * CV_PI / 5.0, 6.0 * CV_PI / 5.0, 8.0 * CV_PI / 5.0};
    
  for (int i = 0, j = 0; i < 5; i++) {
    // 如果在对应的 72度 间隔槽内找到了识别出的扇叶，则填入
    if (j < (int)ts.size() && std::fabs(ts[j].angle - target_angles[i]) < CV_PI / 5.0)
      fanblades.emplace_back(ts[j++]);
    else
      fanblades.emplace_back(FanBlade(_unlight)); // 否则作为未激活占位
  }
};

/**
 * @brief 求极角
 */
double PowerRune::atan_angle(cv::Point2f point) const
{
  auto v = point - r_center;
  auto angle = std::atan2(v.y, v.x);
  return angle >= 0 ? angle : angle + CV_2PI;
}
}  // namespace auto_buff
