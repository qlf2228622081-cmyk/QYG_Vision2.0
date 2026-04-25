#ifndef TOOLS__MATH_TOOLS_HPP
#define TOOLS__MATH_TOOLS_HPP

#include <Eigen/Geometry>
#include <chrono>

namespace tools
{

/**
 * @brief 将弧度值限制在 (-pi, pi] 范围内
 */
double limit_rad(double angle);

/**
 * @brief 四元数转欧拉角
 * @param q 输入四元数
 * @param axis0/1/2 旋转轴索引 (0=X, 1=Y, 2=Z)
 * @param extrinsic 是否为外旋 (静止轴旋转)
 * @return 3D 欧拉角向量
 */
Eigen::Vector3d eulers(
  Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic = false);

/**
 * @brief 旋转矩阵转欧拉角
 */
Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2, bool extrinsic = false);

/**
 * @brief 欧拉角 (Yaw-Pitch-Roll) 转旋转矩阵
 * 顺序：ZYX (先绕 Z 轴 Yaw, 再绕 Y 轴 Pitch, 最后绕 X 轴 Roll)
 */
Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr);

/**
 * @brief 直角坐标系 (XYZ) 转球坐标系 (Yaw, Pitch, Distance)
 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz);

/**
 * @brief 计算 XYZ 转 YPD 的雅可比矩阵 (应用于 EKF 的投影部分)
 */
Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz);

/**
 * @brief 球坐标系 (YPD) 转直角坐标系 (XYZ)
 */
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd);

/**
 * @brief 计算 YPD 转 XYZ 的雅可比矩阵
 */
Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd);

/**
 * @brief 计算两个时间点之间的时间差 (单位：秒)
 */
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b);

/**
 * @brief 计算两个 2D 向量之间的最小夹角
 * @return 范围：[0, pi]
 */
double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2);

/**
 * @brief 平方辅助模板函数
 */
template <typename T>
T square(T const & a)
{
  return a * a;
};

/**
 * @brief 数值限幅函数
 */
double limit_min_max(double input, double min, double max);

}  // namespace tools

#endif  // TOOLS__MATH_TOOLS_HPP