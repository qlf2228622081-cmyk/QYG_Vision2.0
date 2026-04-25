#ifndef AUTO_BUFF__SOLVER_HPP
#define AUTO_BUFF__SOLVER_HPP

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <optional>

#include "buff_type.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff
{

// 能量机关扇叶分布角度
const double THETA = 2.0 * CV_PI / 5.0;  // 72度 (2/5π)

/**
 * @brief 能量机关位姿解算器 (Solver)
 * 逻辑：负责将解析出的像素坐标系下的扇叶/R标，通过 PnP 算法解算出其在相机系、
 * 云台系及世界坐标系（IMU 系）下的 3D 位姿。
 */
class Solver
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，包含相机内参和外参变换矩阵
   */
  explicit Solver(const std::string & config_path);

  /**
   * @brief 获取世界系到云台系的变换矩阵
   */
  Eigen::Matrix3d R_gimbal2world() const;

  /**
   * @brief 设置云台到世界系的变换矩阵（由 IMU 四元数计算得出）
   */
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  /**
   * @brief 执行 PnP 位姿解算
   * @param ps 输入的能量机关识别结果
   */
  void solve(std::optional<PowerRune> & ps) const;

  /**
   * @brief 调试接口：将能量机关局部 3D 点映射回像素点
   */
  cv::Point2f point_buff2pixel(cv::Point3f x);

  /**
   * @brief 调试接口：重投影能量机关所有特征点
   */
  std::vector<cv::Point2f> reproject_buff(
    const Eigen::Vector3d & xyz_in_world, double yaw, double row) const;

private:
  // 相机参数
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  
  // 外参变换矩阵
  Eigen::Matrix3d R_gimbal2imubody_; // 云台到 IMU 载体系
  Eigen::Matrix3d R_camera2gimbal_;  // 相机到云台系
  Eigen::Vector3d t_camera2gimbal_;  // 相机相对于云台的偏移位置
  Eigen::Matrix3d R_gimbal2world_;   // 云台到世界系 (由反馈计算)

  // OpenCV PnP 输出向量
  cv::Vec3d rvec_, tvec_;

  /**
   * @brief 能量机关 3D 模型特征点 (单位：米)
   * 通常是以 R 标中心为原点的各个扇叶角点的物理尺寸。
   */
  const std::vector<cv::Point3f> OBJECT_POINTS = {
    cv::Point3f(0, 0, 827e-3),   // 点 0
    cv::Point3f(0, 127e-3, 700e-3), // 点 1
    cv::Point3f(0, 0, 573e-3),   // 点 2
    cv::Point3f(0, -127e-3, 700e-3),// 点 3
    cv::Point3f(0, 0, 700e-3),   // 扇叶中心 (点 4)
    cv::Point3f(0, 0, 220e-3),   // 方向参考点 (点 5)
    cv::Point3f(0, 0, 0)};       // R 标中心 (点 6)

  /**
   * @brief 生成绕 X 轴旋转的旋转矩阵
   */
  cv::Matx33f rotation_matrix(double angle) const;

  /**
   * @brief 根据旋转角度计算 5 片扇叶的全部物理 3D 点
   */
  void compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points);
};
}  // namespace auto_buff

#endif  // AUTO_AIM__SOLVER_HPP