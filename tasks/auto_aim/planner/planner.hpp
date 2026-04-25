#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{

// MPC 控制周期 (10ms)
constexpr double DT = 0.01;
// 预测时域步数 (Horizon)，总长度 = 2 * HALF_HORIZON
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

/**
 * @brief 轨迹矩阵定义
 * 包含四维状态：yaw, yaw_vel, pitch, pitch_vel，随时间序列展开。
 */
using Trajectory = Eigen::Matrix<double, 4, HORIZON>;

/**
 * @brief 规划结果结构体 (Plan)
 * 包含云台的目标角度、当前应当跟随的角度、角速度及角加速度（用于前馈控制）。
 */
struct Plan
{
  bool control;         // 是否允许云台控制
  bool fire;            // 是否建议发射
  float target_yaw;     // 理想目标 Yaw (不考虑平滑)
  float target_pitch;   // 理想目标 Pitch
  float yaw;            // MPC 规划后的当前 Yaw
  float yaw_vel;        // MPC 规划后的当前 Yaw 角速度
  float yaw_acc;        // MPC 规划后的当前 Yaw 角加速度
  float pitch;          // MPC 规划后的当前 Pitch
  float pitch_vel;      // MPC 规划后的当前 Pitch 角速度
  float pitch_acc;      // MPC 规划后的当前 Pitch 角加速度
};

/**
 * @brief 轨迹规划器 (Planner)
 * 逻辑：基于 TinyMPC 算法，针对预测的目标运动轨迹，生成满足动力学约束（最大加速度等）
 * 的平滑最优控制轨迹。解决了目标快速移动时云台跟随的滞后与振荡问题。
 */
class Planner
{
public:
  Eigen::Vector4d debug_xyza; // 调试观测向量
  
  /**
   * @brief 构造函数
   */
  Planner(const std::string & config_path);

  /**
   * @brief 执行规划逻辑
   * @param target 目标追踪器对象
   * @param bullet_speed 当前射速
   * @return 生成的实时规划路径 Plan
   */
  Plan plan(Target target, double bullet_speed);
  
  /**
   * @brief 带延时补偿的规划接口
   */
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  double yaw_offset_;     // Yaw 标定补偿
  double pitch_offset_;   // Pitch 标定补偿
  double fire_thresh_;    // 允许发射的角度误差阈值
  
  // 不同射速下的预测补偿延时
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;

  // MPC 求解器句柄 (分别对应 Yaw 和 Pitch 轴)
  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  /**
   * @brief 初始化求解器参数（权重 Q/R 矩阵、约束等）
   */
  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  /**
   * @brief 计算单点瞄准目标角度（含弹道补偿）
   */
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);

  /**
   * @brief 在预测时域内生成完整的目标参考轨迹
   */
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP