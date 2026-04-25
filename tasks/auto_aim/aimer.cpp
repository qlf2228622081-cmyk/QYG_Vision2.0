#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_aim
{

/**
 * @brief Aimer 构造函数
 * 从配置文件中加载各项补偿偏置、角度阈值和系统延时等参数。
 */
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  
  // 加载基础偏置角度并转换为弧度
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // 角度转弧度 (180/pi ≈ 57.3)
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // 打击入场角阈值
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // 打击离场角阈值
  
  // 系统延时与速度判定参数
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>(); // 高速下的系统总延迟
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();   // 低速下的系统总延迟
  decision_speed_ = yaml["decision_speed"].as<double>();               // 判断高低速的界限
  
  // 如果定义了左右射击偏移（用于双管或非居中安装结构）
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

/**
 * @brief 主瞄准算法实现
 * 计算从当前时刻到子弹飞行到目标所需时间的预测位置，并考虑弹道下坠。
 */
io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front(); // 默认打击优先级最高的一个目标

  // 获取 EKF 状态向量
  auto ekf = target.ekf();
  
  // 根据目标速度动态选择系统延迟
  double delay_time =
    target.ekf_x()[7] > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  // 基础弹速校准，防止异常输入
  if (bullet_speed < 14) bullet_speed = 23;

  /**
   * 时间预测部分 (Temporal Prediction):
   * 补偿从图像采集时间戳 (timestamp) 到现在 (now) 的处理耗时，
   * 加上额外的系统发射延时 (delay_time)。
   */
  auto future = timestamp;
  if (to_now) {
    double dt;
    // 计算从图像捕获至今的累积耗时 + 设定的系统延时
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future); // 将目标 EKF 状态推算至未来
  }
  else {
    // 固定的估算耗时 (离线测试或简易模式使用)
    auto dt = 0.005 + delay_time;  // 假设检测解算耗时 5ms
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  // 1. 初步选择瞄准点
  auto aim_point0 = choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    return {false, false, 0, 0};
  }

  // 2. 初步解算弹道 (计算飞行时间)
  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  /**
   * 3. 飞行时间迭代优化 (Bullet-Flight-Time Iteration):
   * 瞄准点随时间改变，而弹丸飞行时间又取决于瞄准点的距离。
   * 通过迭代，使预测位置与子弹实际飞抵时刻的位置趋于一致。
   */
  bool converged = false;
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target); // 迭代副本

  for (int iter = 0; iter < 10; ++iter) {
    // 预测未来 = 处理到当前的延迟 + 计算出的子弹飞行时间
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    // 重新根据预测出的状态选择装甲板（主要解决小陀螺状态下的装甲板切换预测）
    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    // 重新解算弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z());

    if (current_traj.unsolvable) {
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // 收敛判断：若飞行时间的变化量小于 1ms，则认为找到了精确解
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      converged = true;
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  /**
   * 4. 计算最终云台角度
   * 结合迭代后的瞄准点和预设的偏移量，合成最终输出。
   */
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = -(current_traj.pitch + pitch_offset_);  // 世界坐标系中 Pitch 向上为负
  return {true, false, yaw, pitch};
}

/**
 * @brief 带有射击模式偏移的瞄准重载
 */
io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  // 根据左右射击模式应用不同的偏移（用于非对称安装的枪管）
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  // 应用当前模式下的具体偏移
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

/**
 * @brief 从多装甲板目标中选择具体击打点
 * 主要解决“小陀螺”旋转过程中，如何平滑地在一块装甲板消失、另一块装甲板出现时进行切换。
 */
AimPoint Aimer::choose_aim_point(const Target & target)
{
  Eigen::VectorXd ekf_x = target.ekf_x(); // 目标 EKF 状态量
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();

  // 如果目标只有一块装甲板被追踪到且未发生预测的跳变
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 1. 计算目标整车中心在世界坐标系下的 Yaw (水平角)
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 2. 计算各装甲板相对于整车中心的夹角偏移
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  /**
   * 情景 1: 非“小陀螺”状态 (速度较慢)
   * 策略：选择最正对我的装甲板，并维持锁定。
   */
  if (std::abs(target.ekf_x()[8]) <= 2 && target.name != ArmorName::outpost) {
    std::vector<int> id_list;
    for (int i = 0; i < armor_num; i++) {
      // 仅考虑正负 60 度范围内的装甲板
      if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
      id_list.push_back(i);
    }
    
    if (id_list.empty()) return {false, armor_xyza_list[0]};

    // 如果有两个或以上的候选（边缘情况），采取锁定策略
    if (id_list.size() > 1) {
      int id0 = id_list[0], id1 = id_list[1];
      // 如果之前没锁定，选择最正对的一个
      if (lock_id_ != id0 && lock_id_ != id1)
        lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;

      return {true, armor_xyza_list[lock_id_]};
    }

    // 只有一个候选时清空锁定状态
    lock_id_ = -1;
    return {true, armor_xyza_list[id_list[0]]};
  }

  /**
   * 情景 2: “小陀螺”旋转状态 (速度较快)
   * 策略：选择正在“进入”视野而不是“离开”视野的装甲板，以提高命中窗口深度。
   */
  double coming_angle, leaving_angle;
  if (target.name == ArmorName::outpost) { // 针对前哨站的特殊旋转逻辑
    coming_angle = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  // 判定旋转方向，选择最优装甲板
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    // 顺时针旋转 vs 逆时针旋转的筛选逻辑
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim
le = 70 / 57.3;
    leaving_angle = 30 / 57.3;
  } else {
    coming_angle = comming_angle_;
    leaving_angle = leaving_angle_;
  }

  // 在小陀螺时，一侧的装甲板不断出现，另一侧的装甲板不断消失，显然前者被打中的概率更高
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > coming_angle) continue;
    if (ekf_x[7] > 0 && delta_angle_list[i] < leaving_angle) return {true, armor_xyza_list[i]};
    if (ekf_x[7] < 0 && delta_angle_list[i] > -leaving_angle) return {true, armor_xyza_list[i]};
  }

  return {false, armor_xyza_list[0]};
}

}  // namespace auto_aim