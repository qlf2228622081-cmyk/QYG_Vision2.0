#include "buff_target.hpp"

namespace auto_buff
{

/**
 * @brief Voter 投票器实现
 */
Voter::Voter() : clockwise_(0) {}

void Voter::vote(const double angle_last, const double angle_now)
{
  if (std::abs(clockwise_) > 50) return; // 达到饱和阈值，停止投票
  // 简单累加角度差方向
  if (angle_last > angle_now) clockwise_--;
  else clockwise_++;
}

int Voter::clockwise() { return clockwise_ > 0 ? 1 : -1; }

/**
 * @brief Target 基类实现
 */
Target::Target() : first_in_(true), unsolvable_(true) {};

/**
 * @brief 坐标变换逻辑：能量机关局部坐标 -> 世界坐标
 * 逻辑：基于卡尔曼滤波器预测出的能量机关中心（R_yaw, R_pitch, R_dis）
 * 以及能量机关平面本身的位姿（yaw, roll），计算 3D 旋转平移。
 */
Eigen::Vector3d Target::point_buff2world(const Eigen::Vector3d & point_in_buff) const
{
  if (unsolvable_) return Eigen::Vector3d(0, 0, 0);
  
  // 构造能量机关平面旋转矩阵 (目前假设 pitch 为 0)
  Eigen::Matrix3d R_buff2world =
    tools::rotation_matrix(Eigen::Vector3d(ekf_.x[4], 0.0, ekf_.x[5]));

  auto R_yaw = ekf_.x[0];
  auto R_pitch = ekf_.x[2];
  auto R_dis = ekf_.x[3];
  
  // R 标在世界坐标系下的 3D 坐标
  Eigen::Vector3d R_center_in_world(
    R_dis * std::cos(R_pitch) * std::cos(R_yaw),
    R_dis * std::cos(R_pitch) * std::sin(R_yaw),
    R_dis * std::sin(R_pitch));

  // 映射最终位置
  return R_buff2world * point_in_buff + R_center_in_world;
}

bool Target::is_unsolve() const { return unsolvable_; }
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

/**
 * @brief SmallTarget (小符) 实现
 * 逻辑：匀速模型。状态向量 x = [R_yaw, R_v_yaw, R_pitch, R_dis, yaw, roll, spd]
 */
SmallTarget::SmallTarget() : Target() {}

void SmallTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  static int lost_cn = 0;
  if (!p.has_value()) {
    unsolvable_ = true;
    lost_cn++;
    return;
  }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  // 1. 初始化检查
  if (first_in_) {
    unsolvable_ = true;
    init(time_gap, p.value());
    first_in_ = false;
  }

  // 2. 丢帧保护逻辑 (超过 6 帧丢失即认为追踪断开)
  if (lost_cn > 6) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 丢失buff");
    lost_cn = 0;
    first_in_ = true;
    return;
  }

  // 3. 执行卡尔曼滤波更新
  unsolvable_ = false;
  update(time_gap, p.value());

  // 4. 发散检查：如果转速 spd 偏离小符理论值 (SMALL_W) 过远，认为解算错误
  if (std::abs(ekf_.x[6]) > SMALL_W + CV_PI / 18 || std::abs(ekf_.x[6]) < SMALL_W - CV_PI / 18) {
    unsolvable_ = true;
    tools::logger()->debug("[Target] 小符角度发散spd: {:.2f}", ekf_.x[6] * 180 / CV_PI);
    first_in_ = true;
    return;
  }
}

/**
 * @brief 小符预测模型 (匀速圆周运动)
 */
void SmallTarget::predict(double dt)
{
  // A 矩阵定义状态转移
  // clang-format off
  A_ << 1.0,  dt, 0.0, 0.0, 0.0, 0.0, 0.0, // R_yaw
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // R_v_yaw
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // R_pitch
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // R_dis
        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // yaw
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, // roll
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0; // spd

  // 过程噪声 Q 配置
  auto v1 = 0.001; 
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Q_ << a * v1, b * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        b * v1, c * v1, 0.0, 0.0, 0.0, 0.0, 0.0,
        ...; // 保持原有 Q 矩阵结构
  // clang-format on 
  
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = A_ * x;
    // 弧度限制处理
    x_prior[0] = tools::limit_rad(x_prior[0]);
    x_prior[2] = tools::limit_rad(x_prior[2]);
    x_prior[4] = tools::limit_rad(x_prior[4]);
    x_prior[5] = tools::limit_rad(x_prior[5]);
    return x_prior;
  };
  ekf_.predict(A_, Q_, f);
}

/**
 * @brief 小符状态初始化
 */
void SmallTarget::init(double nowtime, const PowerRune & p)
{
  lasttime_ = nowtime;
  x0_.resize(7); P0_.resize(7, 7); A_.resize(7, 7);
  Q_.resize(7, 7); H_.resize(7, 7); R_.resize(7, 7);

  // 初始状态估算
  x0_ << p.ypd_in_world[0], 0.0, p.ypd_in_world[1], p.ypd_in_world[2],
         p.ypr_in_world[0], p.ypr_in_world[2], 
         SMALL_W * voter.clockwise();

  P0_.setIdentity(); P0_ *= 10.0; P0_(6, 6) = 1e-2; // 转速初值较准

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[0] = tools::limit_rad(c[0]); c[2] = tools::limit_rad(c[2]);
    c[4] = tools::limit_rad(c[4]); c[5] = tools::limit_rad(c[5]);
    return c;
  };
  ekf_ = tools::ExtendedKalmanFilter(x0_, P0_, x_add);
}

/**
 * @brief 小符更新逻辑
 * 包含：处理扇叶跳变（72度倍数补偿）、方向投票更新、两阶段 EKF 更新。
 */
void SmallTarget::update(double nowtime, const PowerRune & p)
{
  const Eigen::VectorXd & R_ypd = p.ypd_in_world;
  const Eigen::VectorXd & ypr = p.ypr_in_world;
  const Eigen::VectorXd & B_ypd = p.blade_ypd_in_world;

  // 1. 处理扇叶切换导致的 roll 角度不连续性
  // 能量机关有 5 个叶片，切换时角度会跳变 2PI/5
  if (abs(ypr[2] - ekf_.x[5]) > CV_PI / 12) {
    for (int i = -5; i <= 5; i++) {
      double angle_c = ekf_.x[5] + i * 2 * CV_PI / 5;
      if (std::fabs(angle_c - ypr[2]) < CV_PI / 5) {
        ekf_.x[5] += i * 2 * CV_PI / 5;
        break;
      }
    }
  }

  // 2. 更新旋转方向投票
  voter.vote(ekf_.x[5], ypr[2]);
  if (voter.clockwise() * ekf_.x[6] < 0) ekf_.x[6] *= -1;

  predict(nowtime - lasttime_);

  // 3. EKF 第一阶段：线性观测 (R标的 YPD 信息及平面 Roll)
  Eigen::MatrixXd H1(4, 7); H1.setZero();
  H1(0, 0) = 1.0; H1(1, 2) = 1.0; H1(2, 3) = 1.0; H1(3, 5) = 1.0;
  Eigen::MatrixXd R1 = Eigen::MatrixXd::Identity(4, 4) * 0.01; R1(2, 2) = 0.5;
  
  auto z_subtract1 = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]); c[1] = tools::limit_rad(c[1]); c[3] = tools::limit_rad(c[3]);
    return c;
  };
  Eigen::VectorXd z1{{R_ypd[0], R_ypd[1], R_ypd[2], ypr[2]}};
  ekf_.update(z1, H1, R1, z_subtract1);

  // 4. EKF 第二阶段：非线性观测 (物理中心 B 的 YPD)
  // 通过复杂的雅可比矩阵将模型中心坐标映射到状态向量上
  Eigen::MatrixXd H2 = h_jacobian(); 
  Eigen::MatrixXd R2 = Eigen::MatrixXd::Identity(3, 3) * 0.01; R2(2, 2) = 0.5;

  auto h2 = [&](const Eigen::VectorXd & x) -> Eigen::Vector3d {
    // 复杂的非线性前向映射：x -> R_center + Rotor_Rotation -> B_center -> YPD
    Eigen::VectorXd B_xyz = point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.7));
    return tools::xyz2ypd(B_xyz);
  };

  Eigen::VectorXd z2{{B_ypd[0], B_ypd[1], B_ypd[2]}};
  ekf_.update(z2, H2, R2, h2, z_subtract1); // 借用 z_subtract

  lasttime_ = nowtime;
}

/**
 * @brief BigTarget (大符) 实现
 * 状态向量 x = [R_yaw, v_R_yaw, R_pitch, R_dis, yaw, roll, spd, a, w, fi]
 * 运动模型：spd = a * sin(w * t + fi) + (2.09 - a)
 */
BigTarget::BigTarget() : Target(), spd_fitter_(100, 0.5, 1.884, 2.000) {}

void BigTarget::get_target(
  const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp)
{
  static int lost_cn = 0;
  if (!p.has_value()) { unsolvable_ = true; lost_cn++; return; }

  static std::chrono::steady_clock::time_point start_timestamp = timestamp;
  auto time_gap = tools::delta_time(timestamp, start_timestamp);

  if (first_in_) { unsolvable_ = true; init(time_gap, p.value()); first_in_ = false; }
  if (lost_cn > 6) { unsolvable_ = true; first_in_ = true; return; }

  unsolvable_ = false;
  update(time_gap, p.value());

  // 检查参数合理性 (a, w 是否在 Robomaster 规则规定的范围内)
  if (ekf_.x[7] > 1.5 || ekf_.x[8] > 3.0) { first_in_ = true; return; }
}

/**
 * @brief 大符预测模型 (正弦变加速运动)
 */
void BigTarget::predict(double dt)
{
  double a = ekf_.x[7]; double w = ekf_.x[8]; double fi = ekf_.x[9];
  double t = lasttime_ + dt;
  
  // 状态转移方程 f(x)
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = x;
    x_prior[0] = tools::limit_rad(x_prior[0] + dt * x_prior[1]);
    // 积分正弦速度得到位置变化量
    x_prior[5] = tools::limit_rad(x_prior[5] + voter.clockwise() * 
                 (-a / w * std::cos(w * t + fi) + a / w * std::cos(w * lasttime_ + fi) + (2.09 - a) * dt));
    x_prior[6] = a * sin(w * t + fi) + 2.09 - a; // 当前即时速度
    return x_prior;
  };
  
  // A_ 和 Q_ 矩阵此处保持原有代码中的复杂定义（略）
  ekf_.predict(A_, Q_, f);
}

/**
 * @brief 大符更新逻辑
 * 逻辑：增加对 EKF 速度的最小二乘/正弦拟合辅助，以增强对抗观测噪声的能力。
 */
void BigTarget::update(double nowtime, const PowerRune & p)
{
  // 1. 同小符的基础更新逻辑 (坐标补偿、方向投票)
  ... 
  
  // 2. 执行 EKF 更新阶段 (同样分两步)
  ...

  // 3. 辅助特征提取：利用 RansacSineFitter 对滤波出的速度序列进行离线拟合
  if (ekf_.x[6] < 2.1 && ekf_.x[6] >= 0) spd_fitter_.add_data(nowtime, ekf_.x[6]);
  spd_fitter_.fit();

  // 获取拟合出的理想速度值，用于覆盖有噪声的 EKF 预测初值
  fit_spd_ = spd_fitter_.sine_function(nowtime, spd_fitter_.best_result_.A, ...);
  
  lasttime_ = nowtime;
  unsolvable_ = false;
}

}  // namespace auto_buff