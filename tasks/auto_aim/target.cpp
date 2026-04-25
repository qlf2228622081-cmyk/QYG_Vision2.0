#include "target.hpp"

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim {

/**
 * @brief 基于初始观测装甲板构造 Target
 * 流程：通过目标的 yaw（装甲板法线）和半径 r 推算出旋转中心位置，初始化 11 维状态向量。
 */
Target::Target(const Armor &armor, std::chrono::steady_clock::time_point t,
               double radius, int armor_num, Eigen::VectorXd P0_dig)
    : name(armor.name), armor_type(armor.type), jumped(false), last_id(0),
      update_count_(0), armor_num_(armor_num), t_(t), is_switch_(false),
      is_converged_(false), switch_count_(0) {
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd &xyz = armor.xyz_in_world;
  const Eigen::VectorXd &ypr = armor.ypr_in_world;

  // 1. 初步估计旋转中心的坐标
  // 逻辑：中心 = 装甲板位置 + 半径 * cos/sin(航向角)
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // 2. 初始化 11 维状态向量 x0
  // 状态定义：[x, vx, y, vy, z, vz, yaw, v_yaw, r, l, h]
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  
  // 3. 初始协方差矩阵 P0 (对角阵)
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 4. 定义状态加法函数（处理 Yaw 弧度突跳/缠绕问题）
  auto x_add = [](const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  // 5. 初始化扩展卡尔曼滤波器
  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

/**
 * @brief 调试专用初始化
 */
Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4) {
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

/**
 * @brief EKF 预测步 (时间更新)
 */
void Target::predict(std::chrono::steady_clock::time_point t) {
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

/**
 * @brief EKF 预测逻辑
 * 模型：匀速直线运动模型（中心位移） + 匀角速度旋转模型（小陀螺）。
 */
void Target::predict(double dt) {
  // 1. 状态转移矩阵 F (线性部分：位置 -> 位置+v*dt)
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // 2. 过程噪声矩阵 Q
  // 基于分段白噪声模型 (Piecewise White Noise Model)
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;  // 前哨站运动较慢，加速度方差设置较小
    v2 = 0.1; // 角加速度方差
  } else {
    v1 = 90;  // 普通车辆机动性强
    v2 = 400; // 旋转加速快
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 3. 非线性预测函数 (处理航向角限制)
  auto f = [&](const Eigen::VectorXd &x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 4. 特殊车辆模型修正 (前哨站转速强制约束)
  if (this->convergened() && this->name == ArmorName::outpost &&
      std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  ekf_.predict(F, Q, f);
}

/**
 * @brief EKF 更新步 (观测更新)
 * 核心逻辑：解决多装甲板系统的时序关联。
 */
void Target::update(const Armor &armor) {
  // 1. 装甲板匹配：判定观测到的是 4 块（或3块）装甲板中的哪一块
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> &xyza_list = armor_xyza_list();

  // 根据当前 EKF 预测状态，计算出所有可能装甲板的理论位置和角度
  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  // 排序：距离越近的装甲板理论上匹配可能性越高
  std::sort(xyza_i_list.begin(), xyza_i_list.end(),
            [](const std::pair<Eigen::Vector4d, int> &a,
               const std::pair<Eigen::Vector4d, int> &b) {
               Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
               Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
               return ypd1[2] < ypd2[2];
            });

  // 2. 在备选点中寻找角度最匹配的 ID
  for (int i = 0; i < 3 && i < (int)xyza_i_list.size(); i++) {
    const auto &xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error =
        std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
        std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  // 发生装甲板跳变（ID 不为 0）
  if (id != 0) jumped = true;

  // 3. 统计目标切换和更新计数
  if (id != last_id) {
    is_switch_ = true;
    switch_count_++;
  } else {
    is_switch_ = false;
  }

  last_id = id;
  update_count_++;

  // 4. 执行最终的滤波器更新
  update_ypda(armor, id);
}

/**
 * @brief 执行 YPDA (Yaw, Pitch, Distance, Angle) 观测更新
 */
void Target::update_ypda(const Armor &armor, int id) {
  // 1. 构建观测雅可比矩阵 H
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
  
  // 2. 自适应计算测量噪声 R
  // 逻辑： delta_angle 越大（装甲板越斜），测量置信度越低
  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_dig{
      {4e-4, 4e-4, log(std::abs(delta_angle) + 1) + 1,
       log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + 9e-2}};
  Eigen::MatrixXd R = R_dig.asDiagonal();

  // 3. 定义非线性映射函数 h (x -> z)
  auto h = [&](const Eigen::VectorXd &x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  // 4. 处理观测残差计算 (处理弧度缠绕问题)
  auto z_subtract = [](const Eigen::VectorXd &a,
                       const Eigen::VectorXd &b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  // 5. 准备观测向量 z 并更新
  const Eigen::VectorXd &ypd = armor.ypd_in_world;
  const Eigen::VectorXd &ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

/**
 * @brief 返回当前 EKF 状态向量
 */
Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

/**
 * @brief 返回 EKF 实例引用
 */
const tools::ExtendedKalmanFilter &Target::ekf() const { return ekf_; }

/**
 * @brief 获取所有装甲板的列表 (3D 坐标 + Yaw 角度)
 */
std::vector<Eigen::Vector4d> Target::armor_xyza_list() const {
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

/**
 * @brief 直觉判定：检查滤波器输出的物理参数是否合理
 */
bool Target::diverged() const {
  // 检查半径 r 是否在合理范围 (5cm ~ 50cm)
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  // 对于 2 块装甲板的目标（如平衡），检查其长度偏差是否合理
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok)
    return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

/**
 * @brief 判定滤波器是否已收敛
 */
bool Target::convergened() {
  // 普通目标：前 3 帧观测且不发散
  if (this->name != ArmorName::outpost && update_count_ > 3 &&
      !this->diverged()) {
    is_converged_ = true;
  }

  // 前哨站特殊判断 (由于旋转速度大，需要更多观测)
  if (this->name == ArmorName::outpost && update_count_ > 10 &&
      !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

/**
 * @brief 核心物理观测映射函数：基于 ID 将 EKF 中心坐标推导至特定装甲板坐标
 * 逻辑： armor_xyz = center_xyz - [r * cos(angle), r * sin(angle), h]
 */
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd &x, int id) const {
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  
  // 对于某些车辆，相对的两块装甲板具有不同的半径和高度偏置 (l, h)
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

/**
 * @brief 求解观测函数的雅可比矩阵 (H)
 */
Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd &x, int id) const {
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // 1. 部分微分矩阵： xyza 对 11 维状态的偏导
  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  // 2. 第二部分：ypda 对 xyz 坐标的偏导 (利用链式法则)
  Eigen::Vector3d armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
};

bool Target::checkinit() { return isinit; }

} // namespace auto_aim
