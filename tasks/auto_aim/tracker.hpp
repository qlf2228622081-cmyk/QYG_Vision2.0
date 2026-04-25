#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "armor.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{

/**
 * @brief 目标追踪类 (Tracker)
 * 核心逻辑：维护一个状态机，对检测到的装甲板进行时序关联、EKF 处理，
 * 最终输出稳定的 Target 信息。支持多路感知（全向感知）的目标切换。
 */
class Tracker
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，包含追踪参数
   * @param solver 解算器引用，用于更新观测值
   */
  Tracker(const std::string & config_path, Solver & solver);
  Tracker(const std::string & config_path, Solver & solver, std::string enemy_color);

  /**
   * @brief 设置敌方颜色
   */
  void set_enemy_color(const std::string & enemy_color);

  /**
   * @brief 获取当前追踪状态 (lost, detecting, tracking, temp_lost, switching)
   */
  std::string state() const;

  /**
   * @brief 单相机追踪接口
   * @param armors 当前帧检测到的装甲板列表
   * @param t 当前帧时间戳
   * @return std::list<Target> 追踪中的目标列表
   */
  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  /**
   * @brief 多相机/全向感知融合追踪接口
   * 支持感知到视野外更高优先级的目标时，发起目标切换（switching 状态）。
   */
  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;      // 位姿解算器引用
  Color enemy_color_;    // 追踪目标的敌方颜色
  
  // 状态机计数器
  int min_detect_count_;          // 进入 tracking 状态所需的连续帧数
  int max_temp_lost_count_;       // 允许的最大临时丢弃帧数
  int detect_count_;              // 当前检测计数值
  int temp_lost_count_;           // 当前丢失计数值
  int outpost_max_temp_lost_count_; // 前哨站专用的最大容错帧数
  int normal_temp_lost_count_;    // 普通目标最大容错帧数
  
  std::string state_, pre_state_; // 当前和上一帧的状态
  Target target_;                 // 当前追踪的目标对象（内部维护 EKF）
  
  std::chrono::steady_clock::time_point last_timestamp_; // 上一次处理的时间戳
  ArmorPriority omni_target_priority_; // 全向感知锁定的目标优先级

  /**
   * @brief 状态机跳转逻辑
   * 根据当前是否 found 更新 state_ (lost -> detecting -> tracking -> temp_lost ...)
   */
  void state_machine(bool found);

  /**
   * @brief 初始化目标
   * 当状态为 lost 时，尝试从装甲板列表中选出一个合适的作为新目标。
   */
  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  /**
   * @brief 更新现有目标
   * 当处于追踪状态时，执行 EKF 预测并将观测值更新到滤波器中。
   */
  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP