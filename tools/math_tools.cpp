#include "math_tools.cpp"

#include <cmath>
#include <opencv2/core.hpp>

namespace tools
{

/**
 * @brief 弧度限域逻辑 (-pi, pi]
 */
double limit_rad(double angle)
{
  while (angle > CV_PI) angle -= 2 * CV_PI;
  while (angle <= -CV_PI) angle += 2 * CV_PI;
  return angle;
}

/**
 * @brief 复杂的四元数至欧拉角转换实现
 * 算法逻辑：根据旋转顺序（proper/improper）构建中间变量，计算奇异点（万向节锁）并输出弧度。
 */
Eigen::Vector3d eulers(Eigen::Quaterniond q, int axis0, int axis1, int axis2, bool extrinsic)
{
  // 转换逻辑兼容内旋与外旋 (Intrinsic/Extrinsic)
  if (!extrinsic) std::swap(axis0, axis2);

  auto i = axis0, j = axis1, k = axis2;
  auto is_proper = (i == k);
  if (is_proper) k = 3 - i - j;
  auto sign = (i - j) * (j - k) * (k - i) / 2;

  double a, b, c, d;
  Eigen::Vector4d xyzw = q.coeffs();
  if (is_proper) {
    a = xyzw[3]; b = xyzw[i]; c = xyzw[j]; d = xyzw[k] * sign;
  } else {
    a = xyzw[3] - xyzw[j]; b = xyzw[i] + xyzw[k] * sign;
    c = xyzw[j] + xyzw[3]; d = xyzw[k] * sign - xyzw[i];
  }

  Eigen::Vector3d result;
  auto n2 = a * a + b * b + c * c + d * d;
  result[1] = std::acos(2 * (a * a + b * b) / n2 - 1);

  auto half_sum = std::atan2(b, a);
  auto half_diff = std::atan2(-d, c);

  // 奇异点判断逻辑：防止分母为零或失去自由度
  auto eps = 1e-7;
  auto safe1 = std::abs(result[1]) >= eps;
  auto safe2 = std::abs(result[1] - CV_PI) >= eps;
  if (safe1 && safe2) {
    result[0] = half_sum + half_diff;
    result[2] = half_sum - half_diff;
  } else {
    // 处理万向节锁情况 (Gimbal Lock)
    if (!extrinsic) {
      result[0] = 0;
      if (!safe1) result[2] = 2 * half_sum;
      if (!safe2) result[2] = -2 * half_diff;
    } else {
      result[2] = 0;
      if (!safe1) result[0] = 2 * half_sum;
      if (!safe2) result[0] = 2 * half_diff;
    }
  }

  for (int m = 0; m < 3; m++) result[m] = limit_rad(result[m]);

  if (!is_proper) {
    result[2] *= sign;
    result[1] -= CV_PI / 2;
  }

  if (!extrinsic) std::swap(result[0], result[2]);

  return result;
}

/**
 * @brief 欧拉角转旋转矩阵
 * 常规 ZYX 变换实现。
 */
Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr)
{
  double roll = ypr[2]; double pitch = ypr[1]; double yaw = ypr[0];
  double cy = cos(yaw); double sy = sin(yaw);
  double cp = cos(pitch); double sp = sin(pitch);
  double cr = cos(roll); double sr = sin(roll);
  
  Eigen::Matrix3d R;
  R << cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
       sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
       -sp,     cp * sr,                cp * cr;
  return R;
}

/**
 * @brief 直角坐标转球坐标 (XYZ -> YPD)
 */
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  auto yaw = std::atan2(y, x);
  auto pitch = std::atan2(z, std::sqrt(x * x + y * y));
  auto distance = std::sqrt(x * x + y * y + z * z);
  return {yaw, pitch, distance};
}

/**
 * @brief 计算归一化投影雅可比矩阵
 * 用于 EKF 观测方程线性化：将 3D 位置偏差映射为图像/角度偏差。
 */
Eigen::MatrixXd xyz2ypd_jacobian(const Eigen::Vector3d & xyz)
{
  auto x = xyz[0], y = xyz[1], z = xyz[2];
  double r2 = x * x + y * y;
  double r = std::sqrt(r2);
  double R2 = r2 + z * z;
  double R = std::sqrt(R2);

  // 偏导数推导与构建
  auto dyaw_dx = -y / r2;
  auto dyaw_dy = x / r2;
  auto dyaw_dz = 0.0;

  auto dpitch_dx = -(x * z) / (R2 * r);
  auto dpitch_dy = -(y * z) / (R2 * r);
  auto dpitch_dz = r / R2;

  auto ddistance_dx = x / R;
  auto ddistance_dy = y / R;
  auto ddistance_dz = z / R;

  Eigen::MatrixXd J(3, 3);
  J << dyaw_dx, dyaw_dy, dyaw_dz,
       dpitch_dx, dpitch_dy, dpitch_dz,
       ddistance_dx, ddistance_dy, ddistance_dz;
  return J;
}

/**
 * @brief 球坐标转直角坐标 (YPD -> XYZ)
 */
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  return {distance * std::cos(pitch) * std::cos(yaw),
          distance * std::cos(pitch) * std::sin(yaw),
          distance * std::sin(pitch)};
}

/**
 * @brief 正向坐标映射雅可比矩阵
 */
Eigen::MatrixXd ypd2xyz_jacobian(const Eigen::Vector3d & ypd)
{
  auto yaw = ypd[0], pitch = ypd[1], distance = ypd[2];
  double cy = std::cos(yaw); double sy = std::sin(yaw);
  double cp = std::cos(pitch); double sp = std::sin(pitch);

  Eigen::MatrixXd J(3, 3);
  J << -distance * cp * sy, -distance * sp * cy, cp * cy,
        distance * cp * cy, -distance * sp * sy, cp * sy,
        0.0,                 distance * cp,      sp;
  return J;
}

/**
 * @brief 时间管理工具
 */
double delta_time(
  const std::chrono::steady_clock::time_point & a, const std::chrono::steady_clock::time_point & b)
{
  std::chrono::duration<double> c = a - b;
  return c.count();
}

/**
 * @brief 向量余弦相似度计算角度
 */
double get_abs_angle(const Eigen::Vector2d & vec1, const Eigen::Vector2d & vec2)
{
  if (vec1.norm() == 0. || vec2.norm() == 0.) return 0.;
  return std::acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));
}

}  // namespace tools