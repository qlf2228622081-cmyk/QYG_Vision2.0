#include "buff_aimer.cpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff
{

/**
 * @brief Aimer 构造函数
 */
Aimer::Aimer(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  // 加载离线标定偏移量（角度转弧度）
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
  fire_gap_time_ = yaml["fire_gap_time"].as<double>();
  predict_time_ = yaml["predict_time"].as<double>();

  last_fire_t_ = std::chrono::steady_clock::now();
}

/**
 * @brief 传统瞄准控制接口
 * 核心逻辑：
 * 1. 确认目标有效。
 * 2. 计算从图像获取到现在的“过去延时”，叠加系统预估的“预测时长”。
 * 3. 迭代计算弹道飞行时间并更新预测角度。
 * 4. 判定扇叶切换状态，控制开火命令。
 */
io::Command Aimer::aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp,
  double bullet_speed, bool to_now)
{
  io::Command command = {false, false, 0, 0};
  if (target.is_unsolve()) return command;

  // 保护性设定最小射速
  if (bullet_speed < 10) bullet_speed = 24;

  auto now = std::chrono::steady_clock::now();
  
  // 1. 计算预测时长 (Total Latency = Image_Processing_Time + System_Predict_Setting)
  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  
  double yaw, pitch;

  // 2. 调用核心算法获取预测角度
  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    command.yaw = yaw;
    command.pitch = -pitch;  // 弹道解算 Pitch 向上为正，云台指令习惯向上为负，取反
    
    // 3. 目标突变判定：如果目标角度跳变超过 5度，认为目标发生了扇叶切换
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      command.control = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      command.control = false; // 切换瞬间禁止控制，防止云台大幅甩动
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      command.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;
  }

  // 4. 发射时段管理：切换中禁止攻击，平时符合射频（Time Gap）要求则攻击
  if (switch_fanblade_) {
    command.shoot = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    command.shoot = true;
    last_fire_t_ = now;
  }

  return command;
}

/**
 * @brief MPC 模式瞄准接口 ( Model Predictive Control )
 * 增加对角速度 (velocity) 和加速度 (acceleration) 的计算，提供更平滑的规划。
 */
auto_aim::Plan Aimer::mpc_aim(
  auto_buff::Target & target, std::chrono::steady_clock::time_point & timestamp, io::GimbalState gs,
  bool to_now)
{
  auto_aim::Plan plan = {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
  if (target.is_unsolve()) return plan;

  double bullet_speed = (gs.bullet_speed < 10) ? 24 : gs.bullet_speed;
  auto now = std::chrono::steady_clock::now();
  auto detect_now_gap = tools::delta_time(now, timestamp);
  auto future = to_now ? (detect_now_gap + predict_time_) : 0.1 + predict_time_;
  
  double yaw, pitch;

  if (get_send_angle(target, future, bullet_speed, to_now, yaw, pitch)) {
    plan.yaw = yaw;
    plan.pitch = -pitch;
    
    // 处理目标切换逻辑 (同 aim 函数)
    if (mistake_count_ > 3) {
      switch_fanblade_ = true;
      mistake_count_ = 0;
      plan.control = true;
      first_in_aimer_ = true;
    } else if (std::abs(last_yaw_ - yaw) > 5 / 57.3 || std::abs(last_pitch_ - pitch) > 5 / 57.3) {
      switch_fanblade_ = true;
      mistake_count_++;
      plan.control = false;
      first_in_aimer_ = true;
    } else {
      switch_fanblade_ = false;
      mistake_count_ = 0;
      plan.control = true;
    }
    last_yaw_ = yaw;
    last_pitch_ = pitch;

    // 5. MPC 参数计算：计算角速度和角加速度
    if (plan.control) {
      if (first_in_aimer_) {
        plan.yaw_vel = 0; plan.yaw_acc = 0;
        plan.pitch_vel = 0; plan.pitch_acc = 0;
        first_in_aimer_ = false;
      } else {
        auto dt = predict_time_;
        double last_yaw_mpc, last_pitch_mpc;
        // 反推上一帧预计的角度，用差分法估算导数
        get_send_angle(target, predict_time_ * -1, bullet_speed, to_now, last_yaw_mpc, last_pitch_mpc);
        
        // 速度估算：(当前目标 - 上一帧历史位置) / 时间差
        plan.yaw_vel = tools::limit_rad(yaw - last_yaw_mpc) / (2 * dt);
        // 加速度估算：(当前预期偏差 - 历史预期偏差) / dt^2
        plan.yaw_acc = (tools::limit_rad(yaw - gs.yaw) - tools::limit_rad(gs.yaw - last_yaw_mpc)) / std::pow(dt, 2);

        plan.pitch_vel = tools::limit_rad(-pitch + last_pitch_mpc) / (2 * dt);
        plan.pitch_acc = (-pitch - gs.pitch - (gs.pitch + last_pitch_mpc)) / std::pow(dt, 2);
      }
    }
  }

  // 发射逻辑处理 (同 aim 函数)
  if (switch_fanblade_) {
    plan.fire = false;
    last_fire_t_ = now;
  } else if (!switch_fanblade_ && tools::delta_time(now, last_fire_t_) > fire_gap_time_) {
    plan.fire = true;
    last_fire_t_ = now;
  }

  return plan;
}

/**
 * @brief 核心函数：根据弹道学迭代计算应当指向的角度
 * 逻辑迭代流程：
 * 1. 基础预测：先以前馈 predict_time 预测目标旋转到的位置。
 * 2. 第一次弹道解算：根据基础预测位置，计算出子弹在该距离下的飞行时间 t0。
 * 3. 修正预测：根据新的 t0 再次预测目标位置。
 * 4. 第二次弹道解算：根据修正位置计算 t1 和 补偿后的 Pitch 角度。
 * 5. 校验收敛性：如果两次解算的飞行时间差值很小，说明迭代完成。
 */
bool Aimer::get_send_angle(
  auto_buff::Target & target, const double predict_time, const double bullet_speed,
  const bool to_now, double & yaw, double & pitch)
{
  // 1. 执行初始预测
  target.predict(predict_time);

  // 2. 映射 3D 坐标并执行第一次弹道解算
  auto aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7)); // 0.7m 为扇叶相对于 R 标的标准半径
  double d = std::sqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  double h = aim_in_world[2];

  tools::Trajectory trajectory0(bullet_speed, d, h);
  if (trajectory0.unsolvable) return false;

  // 3. 基于飞行时间二次校正预测位置 (迭代)
  target.predict(trajectory0.fly_time);

  aim_in_world = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
  d = fsqrt(aim_in_world[0] * aim_in_world[0] + aim_in_world[1] * aim_in_world[1]);
  h = aim_in_world[2];
  
  tools::Trajectory trajectory1(bullet_speed, d, h);
  if (trajectory1.unsolvable) return false;

  // 4. 收敛性判定：飞行时间变化需小于 10ms
  auto time_error = trajectory1.fly_time - trajectory0.fly_time;
  if (std::abs(time_error) > 0.01) return false;

  // 5. 输出最终 Yaw/Pitch (并叠加离线补偿)
  yaw = std::atan2(aim_in_world[1], aim_in_world[0]) + yaw_offset_;
  pitch = trajectory1.pitch + pitch_offset_;
  return true;
};

}  // namespace auto_buff