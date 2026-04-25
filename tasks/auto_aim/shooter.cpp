#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

/**
 * @brief Shooter 构造函数
 * 从参数配置文件加载开火判定所需的角度门限。
 */
Shooter::Shooter(const std::string & config_path) : last_command_{false, false, 0, 0}
{
  auto yaml = YAML::LoadFile(config_path);
  // 将角度值转换为弧度值
  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();
}

/**
 * @brief 执行射击判定逻辑
 * 核心逻辑：
 * 1. 检查是否开启自动开火且有追踪目标。
 * 2. 根据目标距离动态切换容忍度门限（远距离更严格，近距离较宽松）。
 * 3. 检查当前云台指向与预测目标点的偏差是否在容忍范围内。
 * 4. 防止指令突变（跳变）导致的误射。
 */
bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  // 1. 基础前置条件过滤
  if (!command.control || targets.empty() || !auto_fire_) return false;

  // 2. 根据目标距离决定开火精度要求 (Tolerance)
  auto target_x = targets.front().ekf_x()[0];
  auto target_y = targets.front().ekf_x()[2];
  auto distance = std::sqrt(tools::square(target_x) + tools::square(target_y));
  
  auto tolerance = distance > judge_distance_
                     ? second_tolerance_  // 远距离门限
                     : first_tolerance_;   // 近距离门限

  // 3. 综合判定：
  //    a) 角度差值：当前云台角度与下发指令角度的偏差小于门限；
  //    b) 指令连续性：防止目标瞬间切换导致的误射；
  //    c) 弹道有效性：Aimer 必须已经算出了合法的瞄点。
  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 && // 抑制指令突变
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&   // 云台基本对准
    aimer.debug_aim_point.valid) {
    
    last_command_ = command;
    return true;
  }

  // 4. 更新上一帧指令缓存并返回否定结果
  last_command_ = command;
  return false;
}

}  // namespace auto_aim