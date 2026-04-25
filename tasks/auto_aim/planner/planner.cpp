#include "planner.cpp"

#include <vector>

#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{

/**
 * @brief Planner 构造函数
 * 逻辑：加载 MPC 权重参数并初始化 Yaw/Pitch 分轴求解器。
 */
Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  // 加载标定与控制参数
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

/**
 * @brief 核心规划逻辑 (MPC Solve)
 * 步骤：
 * 1. 估算初始飞行时间并执行粗略目标预测。
 * 2. 生成未来 HORIZON 长度内的参考运动轨迹 (X_ref)。
 * 3. 将当前云台状态设为 x0。
 * 4. 调用 TinyMPC 求解最优控制序列 (u)。
 * 5. 提取规划时域中间位置的状态作为当前指令下发。
 */
Plan Planner::plan(Target target, double bullet_speed)
{
  // 1. 获取目标最近装甲板的飞行时间初值
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  // 执行基于飞行时间的预瞄预测
  target.predict(bullet_traj.fly_time);

  // 2. 生成预测时域内的完整参考轨迹
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0); // 以当前预测点的 Yaw 为基准基底
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. 求解 Yaw 轴 MPC
  Eigen::VectorXd x0_yaw(2);
  x0_yaw << traj(0, 0), traj(1, 0); // 初始点状态
  tiny_set_x0(yaw_solver_, x0_yaw);
  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON); // 设置参考轨迹
  tiny_solve(yaw_solver_); // 执行 QP 优化

  // 4. 求解 Pitch 轴 MPC
  Eigen::VectorXd x0_pitch(2);
  x0_pitch << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0_pitch);
  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  // 5. 构造指令 Plan
  Plan plan;
  plan.control = true;
  
  // 取预瞄时域中间点作为下发目标 (HALF_HORIZON)，增加平滑度
  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON); // 提取角速度前馈
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON); // 提取最优加速度

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  // 6. 开火判定：检查规划轨迹与目标实际轨迹的误差是否在阈值内
  auto shoot_offset_ = 2; // 向后偏移几帧检查，确保云台已经稳定跟随
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
        
  return plan;
}

/**
 * @brief 带系统延时补偿的规划入口
 */
Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};
  
  // 动态补偿时间：根据目标转速决定，高速下需要更长的提前量
  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));
  target->predict(future);

  return plan(*target, bullet_speed);
}

/**
 * @brief 初始化 Yaw 求解器配置
 */
void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  
  // 离群点权重控制：Q 决定跟随精度，R 决定控制量平滑度
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  // 离散化二阶线性模型：x(k+1) = Ax(k) + Bu(k)
  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), ...);

  // 设置约束边界 (State bound constraints & Input bound constraints)
  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  yaw_solver_->settings->max_iter = 10; // 固定迭代次数，保证实时性 10ms
}

/**
 * @brief 初始化 Pitch 求解器配置
 */
void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, ...)

  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, ...);

  pitch_solver_->settings->max_iter = 10;
}

/**
 * @brief 单点瞄准计算逻辑
 */
Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  // 选取最近的装甲板作为目标点
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
    }
  }
  
  // 计算观测偏航角并补偿弹道 Pitch
  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  
  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

/**
 * @brief 在时域内生成参考轨迹 (X_ref)
 * 逻辑：在当前时间点前后各预测一段路径，通过差分推算速度，最终构建出完整的 X_ref 状态矩阵。
 */
Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  // 1. 初始化起始状态 (推算到 HALF_HORIZON 之前)
  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed);

  // 2. 循环预测未来序列
  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    // 计算即时角速度 (Yaw 改为相对于基准 yaw0 的相对量)
    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    // 填充参考轨迹列
    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim