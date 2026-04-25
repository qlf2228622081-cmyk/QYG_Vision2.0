#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

#include "io/cboard.hpp"
#include "io/command.hpp"
#include "target.hpp"

namespace auto_aim
{

/**
 * @brief 瞄准点结构体
 * 包含预测出的目标三维坐标及有效性标志
 */
struct AimPoint
{
  bool valid;           // 瞄准点是否有效
  Eigen::Vector4d xyza; // 坐标 (x, y, z) 以及可能的额外属性 (如角度 a)
};

/**
 * @brief 决策与预测类 (Aimer)
 * 负责根据追踪器的目标状态，结合弹道补偿、时间延迟等因素，计算最终的云台控制指令。
 */
class Aimer
{
public:
  AimPoint debug_aim_point; // 用于调试的瞄准点信息

  /**
   * @brief 构造函数，加载配置文件
   * @param config_path 配置文件路径
   */
  explicit Aimer(const std::string & config_path);

  /**
   * @brief 计算瞄准指令
   * @param targets 追踪到的目标列表
   * @param timestamp 图像的时间戳
   * @param bullet_speed 当前弹速
   * @param to_now 是否补偿到当前时刻 (实战建议 true)
   * @return io::Command 包含 yaw, pitch 和有效性标志的控制指令
   */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  /**
   * @brief 计算瞄准指令 (支持射击模式偏移)
   * @param targets 追踪到的目标列表
   * @param timestamp 图像的时间戳
   * @param bullet_speed 当前弹速
   * @param shoot_mode 射击模式 (左/右射击偏移)
   * @param to_now 是否补偿到当前时刻
   * @return io::Command 控制指令
   */
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  // 基础偏移量 (弧度)
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;

  // 目标筛选角度阈值
  double comming_angle_; // 目标切入视角阈值
  double leaving_angle_; // 目标离开视角阈值

  // 锁定机制
  double lock_id_ = -1;  // 当前锁定装甲板的内部 ID

  // 延时补偿参数 (秒)
  double high_speed_delay_time_; // 高速移动时的系统延时补偿
  double low_speed_delay_time_;  // 低速移动时的系统延时补偿
  double decision_speed_;        // 速度判别阈值

  /**
   * @brief 从目标候选装甲板中选择最适合击打的一块
   * @param target 目标对象
   * @return AimPoint 选择出的瞄准点信息
   */
  AimPoint choose_aim_point(const Target & target);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP