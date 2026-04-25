#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
// 装甲板物理尺寸常量 (单位：米)
constexpr double LIGHTBAR_LENGTH = 56e-3;     // 灯条长度 (56mm)
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // 大装甲板宽度 (230mm)
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // 小装甲板宽度 (135mm)

// 大装甲板 3D 模型点云（世界系/目标系中心坐标）
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

// 小装甲板 3D 模型点云
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

/**
 * @brief Solver 构造函数
 * 从静态参数文件中加载相机内参 (camera_matrix) 和各坐标系间的固定变换。
 */
Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  // 1. 加载姿态矩阵 (Gimbal <-> IMU <-> Camera)
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // 2. 加载相机内参和畸变系数，并转换为 OpenCV 格式
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

/**
 * @brief 获取云台旋转矩阵
 */
Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

/**
 * @brief 更新外部传入的云台位姿（通常来自控制板 IMU）
 * @param q 云台位姿四元数
 * 逻辑：将 IMU 数据映射到云台坐标系中。
 */
void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  // 变换链：R_gimbal -> R_imubody -> R_imuabs -> R_world
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

/**
 * @brief 解算装甲板三维位姿 (Core Solve Function)
 * 流程：solvePnP -> 坐标系转换 (Camera -> Gimbal -> World) -> 最终姿态计算
 */
void Solver::solve(Armor & armor) const
{
  // 1. 选择对应的 3D 模型点
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  // 2. 解 PnP：根据 2D 像素坐标和 3D 模型点求取相机系下的平移/旋转
  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  // 3. 坐标系转换：Camera -> Gimbal -> World
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  // 4. 计算旋转 (Yaw/Pitch/Roll)
  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  
  // 存储相对于云台和世界系的角度
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  // 转换为球坐标系 (Yaw, Pitch, Distance)
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 对于“平衡”步兵，由于 Pitch 假设不成立，通常跳过 Yaw 优化
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  // 5. 进行 Yaw 角优化（基于重投影误差）
  optimize_yaw(armor);
}

/**
 * @brief 将世界系下的装甲板重投影回像素平面
 */
std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  // 大部分装甲板有 15 度的结构倾角，前哨站反之
  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // 构建装甲板到世界的旋转矩阵
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };

  // 转换链：World -> Gimbal -> Camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // 应用重投影
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

/**
 * @brief 计算前哨站专用的重投影误差
 */
double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };

  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * xyz_in_world - t_camera2gimbal_);

  std::vector<cv::Point2f> image_points;
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

/**
 * @brief Yaw 角度优化逻辑 (Iterative Yaw Optimization)
 * 在 [-70, 70] 度范围内进行暴力搜索，寻找使重投影误差最小的 Yaw。
 * 效果：显著提高 PnP 求得的斜侧面装甲板角度的准确性。
 */
void Solver::optimize_yaw(Armor & armor) const
{
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // 搜索范围：140 度
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    // 计算各个测试角度下的像素偏差
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  // 保存原始 Yaw 用于对照，并更新为优化后的最优 Yaw
  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

/**
 * @brief 高级成本函数 (SJTU Cost)
 * 结合了角点偏差和灯条斜率偏差的加权误差模型。
 */
double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs, pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    Eigen::Vector2d ref_d = refs[p] - refs[i]; 
    Eigen::Vector2d pt_d = pts[p] - pts[i];
    
    // 1. 距离代价 (Pixel distance)
    double pixel_dis =
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    
    // 2. 角度/斜率代价 (Angular distance)
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    
    // 3. 动态权重合成 (根据目标倾斜度调整相信角度还是相信距离)
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0; 
    
    cost += std::sqrt(cost_i);
  }
  return cost;
}

/**
 * @brief 计算重投影误差辅助函数
 */
double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  // 计算 4 个角点的像素距离误差累加
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

/**
 * @brief 世界坐标到像素坐标的转换 (辅助工具)
 */
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec, tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    // 检查点是否在相机前方
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;
    if (camera_point.z() > 0) valid_world_points.push_back(world_point);
  }
  
  if (valid_world_points.empty()) return {};

  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}

}  // namespace auto_aim