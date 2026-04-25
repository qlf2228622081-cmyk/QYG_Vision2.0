#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

/**
 * @brief 追踪目标类 (Target)
 * 该类封装了针对“小陀螺”（旋转目标）的运动学模型。
 * 通过扩展卡尔曼滤波器 (EKF) 维护目标的中心位置、速度、旋转角、角速度以及半径等 11 维状态。
 */
class Target
{
public:
  ArmorName name;        // 目标兵种名称
  ArmorType armor_type;  // 装甲板类型（大小）
  ArmorPriority priority; // 优先级
  bool jumped;           // 是否发生了装甲板跳变（切换了观测的装甲板）
  int last_id;           // 上一次观测的装甲板编号

  Target() = default;
  
  /**
   * @brief 构造函数：基于初始观测装甲板初始化目标
   * @param armor 初始探测到的装甲板
   * @param t 初始时间戳
   * @param radius 预设初始半径
   * @param armor_num 目标装甲板总数（通常为 2, 3, 4）
   * @param P0_dig 初始协方差对角阵元素
   */
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);

  /**
   * @brief 调试专用构造函数
   */
  Target(double x, double vyaw, double radius, double h);

  /**
   * @brief 滤波器预测步 (时间更新)
   */
  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  /**
   * @brief 滤波器更新步 (观测更新)
   * 包含装甲板时序匹配逻辑（判定观测到了哪一块装甲板）。
   */
  void update(const Armor & armor);

  // Getter 接口
  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  
  /**
   * @brief 获取所有装甲板在世界系下的 3D 坐标列表
   * 基于当前中心位置和旋转角推算出所有潜在装甲板的位置。
   */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  /**
   * @brief 滤波器发散判定
   * 检查状态量（如半径）是否处于物理合理范围内。
   */
  bool diverged() const;

  /**
   * @brief 滤波器收敛判定
   */
  bool convergened();

  bool isinit = false; // 用于外部初始化的标志位
  bool checkinit();    // 检查是否初始化完成

private:
  int armor_num_;   // 装甲板数量 (极坐标模型分片数)
  int switch_count_; // 装甲板切换计数
  int update_count_; // 观测更新总计数

  bool is_switch_, is_converged_;

  /**
   * @brief 扩展卡尔曼滤波器实例
   * 11 维状态向量说明 (x):
   * [0] center_x, [1] center_vx  -- 旋转中心 X 轴位置及速度
   * [2] center_y, [3] center_vy  -- 旋转中心 Y 轴位置及速度
   * [4] center_z, [5] center_vz  -- 旋转中心 Z 轴位置及速度
   * [6] yaw,      [7] v_yaw      -- 当前主装甲板航向角及角速度
   * [8] r,        [9] l          -- 半径 (r为主半径, l为轴向偏差)
   * [10] h                       -- 高度差 (z轴偏差)
   */
  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_; // 记录最后一次更新的时间

  /**
   * @brief 观测更新核心接口
   */
  void update_ypda(const Armor & armor, int id);  // yaw pitch distance angle

  /**
   * @brief 观测函数 (h)：从状态向量转化为 3D 空间坐标
   */
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  /**
   * @brief 观测雅可比矩阵计算 (H)
   */
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP