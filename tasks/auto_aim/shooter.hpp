#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"

namespace auto_aim
{

/**
 * @brief 发射控制器 (Shooter)
 * 逻辑：基于当前云台姿态与下发的控制指令（Command）之间的偏差，
 * 判定是否达到了发射条件（开火门限控制）。
 */
class Shooter
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，加载开火门限参数
   */
  Shooter(const std::string & config_path);

  /**
   * @brief 判定是否执行发射
   * @param command 即将发给电控的控制指令
   * @param aimer 瞄准器实例（包含弹道预测状态）
   * @param targets 当前追踪的目标列表
   * @param gimbal_pos 云台当前实时位姿 (反馈值)
   * @return true: 允许射击; false: 禁止射击
   */
  bool shoot(
    const io::Command & command, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
  io::Command last_command_; // 记录上一帧的控制指令
  double judge_distance_;    // 远近距离切换门限
  double first_tolerance_;   // 近距离下的开火误差容忍度 (rad)
  double second_tolerance_;  // 远距离下的开火误差容忍度 (rad)
  bool auto_fire_;           // 是否开启自动开火开关
};
}  // namespace auto_aim

#endif  // AUTO_AIM__SHOOTER_HPP