#ifndef AUTO_BUFF__AIMER_HPP
#define AUTO_BUFF__AIMER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <vector>

#include "../auto_aim/planner/planner.hpp"
#include "buff_target.hpp"
#include "buff_type.hpp"
#include "io/command.hpp"
#include "io/gimbal/gimbal.hpp"

namespace auto_buff
{

/**
 * @brief 能量机关瞄准器 (Aimer)
 * 逻辑：基于识别出的能量机关实时状态（Target），结合弹道学模型和预测时间，
 * 计算出应当下发给云台的 Yaw/Pitch 指令及发射信号。
 * 支持传统 PID 指令模式和现代 MPC (Model Predictive Control) 轨迹规划模式。
 */
class Aimer
{
public:
  /**
   * @brief 构造函数
   */
  Aimer(const std::string & config_path);

  /**
   * @brief 传统瞄准接口 (输出单帧 Command)
   * @param target 已解算的能量机关目标
   * @param timestamp 图像原始时间戳
   * @param bullet_speed 当前射速 (m/s)
   * @param to_now 是否预测到当前时刻 (用于补偿系统延迟)
   */
  io::Command aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, double bullet_speed,
    bool to_now = true);

  /**
   * @brief 基于 MPC 的先进瞄准接口 (输出含导数的 Plan)
   */
  auto_aim::Plan mpc_aim(
    Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
    bool to_now = true);

  double angle;      // 目标当前旋转角度 (调试观测)
  double t_gap = 0;  // 弹道飞行时间偏差 (调试观测)

private:
  SmallTarget target_;    // 内部目标模型
  double yaw_offset_;     // 离线标定的 Yaw 补偿 (rad)
  double pitch_offset_;   // 离线标定的 Pitch 补偿 (rad)

  double fire_gap_time_;  // 两发子弹之间的最小间隔 (控制射速)
  double predict_time_;   // 系统预设的前馈预测时长 (s)

  int mistake_count_ = 0; // 识别跳变计数器，用于防止目标瞬间切换导致的误射
  bool switch_fanblade_;  // 是否正在切换扇叶 (切换期间禁止开火)

  double last_yaw_ = 0;   // 上一帧计算的 Yaw 指令
  double last_pitch_ = 0; // 上一帧计算的 Pitch 指令

  // MPC 相关状态变量
  bool first_in_aimer_ = true;
  std::chrono::steady_clock::time_point last_fire_t_;

  /**
   * @brief 核心算法：计算最终发送给云台的角度 (含弹道下坠补偿)
   */
  bool get_send_angle(
    auto_buff::Target & target, const double predict_time, const double bullet_speed,
    const bool to_now, double & yaw, double & pitch);
};
}  // namespace auto_buff

#endif  // AUTO_AIM__AIMER_HPP