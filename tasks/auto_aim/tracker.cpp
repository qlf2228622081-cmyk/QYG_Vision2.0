#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

/**
 * @brief Tracker 构造函数
 * 初始化状态为 lost，并从配置文件中加载状态机阈值。
 */
 Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth} // 默认最低优先级
{
  auto yaml = YAML::LoadFile(config_path);
  // 加载状态跳变所需的计数阈值
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

/**
 * @brief 设置目标敌方颜色
 */
void Tracker::set_enemy_color(const std::string & enemy_color)
{
  enemy_color_ = (enemy_color == "red") ? Color::red : Color::blue;
}

/**
 * @brief 获取当前追踪状态
 */
std::string Tracker::state() const { return state_; }

/**
 * @brief 装甲板追踪逻辑 (单相机)
 * 流程：颜色过滤 -> 排序 -> 状态更新 (lost -> set, tracking -> update) -> 状态机更新 -> 异常检查
 */
std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  // 1. 计算时间间隔 dt
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机掉线或跳帧，重置为丢失状态
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 2. 预处理：过滤非我方颜色目标
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 3. 排序逻辑：
  // 3.1 优先选择靠近图像中心的目标 (减少云台转动负担)
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // 默认 1080p 中心
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 3.2 按照兵种优先级排序 (如英雄优先级高于步兵)
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  // 4. 执行核心追踪逻辑：丢失则尝试初始化，非丢失则尝试更新
  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  } else {
    found = update_target(armors, t);
  }

  // 5. 更新状态机计数并维护状态跳转
  state_machine(found);

  // 6. 异常判定
  // 6.1 滤波器发散检测 (Prediction error too large)
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 6.2 NIS (Normalized Innovation Squared) 一致性检验：
  // 如果近期 NIS 失败比例过高，说明模型与观测严重不符，判定为追踪失败。
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  return {target_};
}

/**
 * @brief 融合追踪接口 (支持主相机 + 全向感知相机)
 * 支持多相机之间的优先级抢占机制 (Preemption)
 */
std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 主相机目标排序 (同上)
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }
  // 抢占逻辑 A：主相机内出现了比当前目标优先级更高的装甲板
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }
  // 抢占逻辑 B：全向感知相机探测到了更重要的目标 (进入 switching 状态，通知系统大范围转弯)
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }
  // 处理 switching 过程中的目标关联
  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }
  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }
  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  state_machine(found);

  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};
  }

  if (state_ == "lost") return {switch_target, {}};

  return {switch_target, {target_}};
}

/**
 * @brief 追踪器内部计数状态机
 * lost: 彻底丢失目标
 * detecting: 连续发现几帧，即将进入稳定追踪
 * tracking: 稳定追踪中（EKF 已收敛）
 * temp_lost: 临时丢失（允许几帧容错，由预测维持）
 * switching: 主动强制切换目标中
 */
void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;
    state_ = "detecting";
    detect_count_ = 1;
  }
  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }
  else if (state_ == "tracking") {
    if (found) return;
    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }
  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost"; // 切换过程中允许较长容错
    }
  }
  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      // 对于旋转的前哨站，丢失容错帧数可以单独设置更长
      if (target_.name == ArmorName::outpost)
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

/**
 * @brief 初始化 Target 对象并配置 EKF 初始协方差
 * 针对不同兵种（平衡步兵、前哨站、基地等）配置不同的过程噪声与初始偏差。
 */
bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor); // 首先解算 3D 位姿

  // 判定是否为“平衡”步兵（通常 3/4/5 号车有大平衡底盘）
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    // 平衡底盘参数：半径较宽 (0.2m)，3块装甲板
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }
  else if (armor.name == ArmorName::outpost) {
    // 前哨站参数：固定中心，半径较大 (0.2765m)，3块装甲板，旋转中心不随车体移动
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }
  else if (armor.name == ArmorName::base) {
    // 基地参数：大型目标 (0.3205m)
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }
  else {
    // 普通步兵/英雄参数
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

/**
 * @brief 执行 EKF 预测与更新步
 */
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  // 1. 根据运动模型进行预测
  target_.predict(t);

  // 2. 在当前检测结果中寻找与 Tracker 目标匹配（ID 和 Type 一致）的最佳观测值
  int found_count = 0;
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
  }

  if (found_count == 0) return false;

  // 3. 执行更新步：目前简单匹配第一个符合名称的目标。
  // TODO: 后续可加入位置距离重匹配逻辑 (Association Logic)
  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;

    solver_.solve(armor); // 解算最新位姿
    target_.update(armor); // 送入 EKF 滤波器更新状态
    break; // 仅更新一个观测值
  }

  return true;
}

}  // namespace auto_aim