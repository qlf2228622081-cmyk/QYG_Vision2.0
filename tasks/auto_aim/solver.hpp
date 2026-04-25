#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief 位姿解算类 (Solver)
 * 负责将相机系下的装甲板检测结果转化为云台系、世界系下的坐标，
 * 涉及解 PnP、重投影、坐标系变换及 Yaw 角度优化。
 */
class Solver
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，包含相机内参及相机-云台外参
   */
  explicit Solver(const std::string & config_path);

  /**
   * @brief 获取当前云台到世界系的旋转矩阵
   */
  Eigen::Matrix3d R_gimbal2world() const;

  /**
   * @brief 设置云台到世界系的变换关系 (通常由 IMU 反馈的四元数得到)
   * @param q 云台相对于大地的位姿四元数
   */
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  /**
   * @brief 解算装甲板位姿
   * 通过 solvePnP 得到相对于相机的位姿，并转换到云台系和世界系。
   * @param armor 输入检测到的装甲板，输出包含位姿信息的装甲板
   */
  void solve(Armor & armor) const;

  /**
   * @brief 装甲板重投影
   * 将世界系坐标映射回图像像素坐标，常用于验证位姿解算的正确性。
   * @param xyz_in_world 世界系下的 3D 坐标
   * @param yaw 目标当前的航向角
   * @param type 装甲板类型（大小装甲）
   * @param name 装甲板名称
   * @return std::vector<cv::Point2f> 投影后的像素坐标列表
   */
  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  /**
   * @brief 计算前哨站重投影误差 (专用于特殊倾斜旋转的目标)
   */
  double oupost_reprojection_error(Armor armor, const double & picth);

  /**
   * @brief 通用的世界坐标到像素坐标转换
   * @param worldPoints 世界系 3D 点云
   * @return std::vector<cv::Point2f> 像素系 2D 点云
   */
  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  cv::Mat camera_matrix_;     // 相机内参矩阵
  cv::Mat distort_coeffs_;    // 相机畸变系数
  
  Eigen::Matrix3d R_gimbal2imubody_; // 云台到 IMU 载体系的固定旋转
  Eigen::Matrix3d R_camera2gimbal_;  // 相机到云台的固定旋转 (外参)
  Eigen::Vector3d t_camera2gimbal_;  // 相机到云台的固定平移 (外参)
  Eigen::Matrix3d R_gimbal2world_;   // 云台到世界系的实时旋转矩阵

  /**
   * @brief 优化装甲板的 Yaw 角
   * solvePnP 得到的旋转在平面上往往不够准，通过重投影误差最小化来修正。
   */
  void optimize_yaw(Armor & armor) const;

  /**
   * @brief 计算装甲板重投影误差
   */
  double armor_reprojection_error(const Armor & armor, double yaw, const double & inclined) const;
  
  /**
   * @brief SJTU 风格的 Cost 计算函数 (结合像素距离与角度距离)
   */
  double SJTU_cost(
    const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
    const double & inclined) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP