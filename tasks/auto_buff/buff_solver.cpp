#include "buff_solver.hpp"

namespace auto_buff
{

/**
 * @brief 生成旋转矩阵
 */
cv::Matx33f Solver::rotation_matrix(double angle) const
{
  return cv::Matx33f(
    1, 0, 0, 0, std::cos(angle), -std::sin(angle), 0, std::sin(angle), std::cos(angle));
}

/**
 * @brief 计算所有扇叶在模型坐标系下的 3D 点
 */
void Solver::compute_rotated_points(std::vector<std::vector<cv::Point3f>> & object_points)
{
  const std::vector<cv::Point3f> & base_points = object_points[0];
  for (int i = 1; i < 5; ++i) {
    double angle = i * THETA;
    cv::Matx33f R = rotation_matrix(angle);
    std::vector<cv::Point3f> rotated_points;
    for (const auto & point : base_points) {
      cv::Vec3f vec(point.x, point.y, point.z);
      cv::Vec3f rotated_vec = R * vec;
      rotated_points.emplace_back(rotated_vec[0], rotated_vec[1], rotated_vec[2]);
    }
    object_points[i] = rotated_points;
  }
}

/**
 * @brief Solver 构造函数
 * 初始化相机参数和外参矩阵。
 */
Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  // 加载外参 (相机 -> 云台 -> IMU)
  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  // 加载内参
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

/**
 * @brief 设置云台到世界系的坐标转换矩阵
 * 逻辑：基于 IMU 主板反馈的四元数 q 计算 IMU Body 到世界系的旋转，
 * 然后结合云台外参计算出云台整体在世界系下的位姿。
 */
void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  // 变换：Gimbal -> IMU_Body -> IMU_Abs(World) -> Gimbal_Abs
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

/**
 * @brief 执行位姿解算
 * 逻辑：
 * 1. 选取目标扇叶的 4 个关键点。
 * 2. 结合 R 标中心点构成 PnP 解算点对。
 * 3. 调用 solvePnP (IPPE 算法处理平面目标) 解算出目标在相机系下的位置。
 * 4. 执行坐标系链式变换：相机系 -> 云台系 -> 世界系。
 */
void Solver::solve(std::optional<PowerRune> & ps) const
{
  if (!ps.has_value()) return;
  PowerRune & p = ps.value();

  // 1. 构造 2D-3D 匹配点
  std::vector<cv::Point2f> image_points = p.target().points; // 扇叶回归关键点
  image_points.emplace_back(p.r_center); // R 标中心

  // 使用前四个角点执行快速 PnP
  std::vector<cv::Point2f> image_points_fourth(image_points.begin(), image_points.begin() + 4);
  std::vector<cv::Point3f> OBJECT_POINTS_FOURTH(OBJECT_POINTS.begin(), OBJECT_POINTS.begin() + 4);
  
  // 执行解算
  cv::solvePnP(
    OBJECT_POINTS_FOURTH, image_points_fourth, camera_matrix_, distort_coeffs_, rvec_, tvec_, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d t_buff2camera;
  cv::cv2eigen(tvec_, t_buff2camera);
  cv::Mat rmat;
  cv::Rodrigues(rvec_, rmat);
  Eigen::Matrix3d R_buff2camera;
  cv::cv2eigen(rmat, R_buff2camera);

  // 2. 能量机关局部物理点 (中心位置 z=0.7)
  Eigen::Vector3d blade_xyz_in_buff{{0, 0, 700e-3}};

  // 3. 坐标转换：Buff -> Camera
  Eigen::Vector3d xyz_in_camera = t_buff2camera; // R 标在相机系
  Eigen::Vector3d blade_xyz_in_camera = R_buff2camera * blade_xyz_in_buff + t_buff2camera;

  // 4. 坐标转换：Camera -> Gimbal
  Eigen::Matrix3d R_buff2gimbal = R_camera2gimbal_ * R_buff2camera;
  Eigen::Vector3d xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  Eigen::Vector3d blade_xyz_in_gimbal = R_camera2gimbal_ * blade_xyz_in_camera + t_camera2gimbal_;

  // 5. 坐标转换：Gimbal -> World
  Eigen::Matrix3d R_buff2world = R_gimbal2world_ * R_buff2gimbal;

  // 存储最终解算结果 (R 标世界坐标)
  p.xyz_in_world = R_gimbal2world_ * xyz_in_gimbal;
  p.ypd_in_world = tools::xyz2ypd(p.xyz_in_world);

  // 存储最终解算结果 (目标扇叶世界坐标)
  p.blade_xyz_in_world = R_gimbal2world_ * blade_xyz_in_gimbal;
  p.blade_ypd_in_world = tools::xyz2ypd(p.blade_xyz_in_world);

  // 解析姿态角 (Yaw, Pitch, Roll)
  p.ypr_in_world = tools::eulers(R_buff2world, 2, 1, 0);
}

/**
 * @brief 调试：将 Buff 系下的点映射回图像像素
 */
cv::Point2f Solver::point_buff2pixel(cv::Point3f x)
{
  std::vector<cv::Point3d> world_points = {x};
  std::vector<cv::Point2d> image_points;
  cv::projectPoints(world_points, rvec_, tvec_, camera_matrix_, distort_coeffs_, image_points);
  return image_points.back();
}

/**
 * @brief 调试：重投影整个能量机关模型
 * 逻辑：基于已知的世界坐标和位姿，反推回相机系并投影到像素坐标。用于验证位姿解算准确性。
 */
std::vector<cv::Point2f> Solver::reproject_buff(
  const Eigen::Vector3d & xyz_in_world, double yaw, double row) const
{
  auto R_buff2world = tools::rotation_matrix(Eigen::Vector3d(yaw, 0.0, row));

  // 获取世界系到相机系的变换关系
  const Eigen::Vector3d & t_buff2world = xyz_in_world;
  Eigen::Matrix3d R_buff2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_buff2world;
  Eigen::Vector3d t_buff2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_buff2world - t_camera2gimbal_);

  cv::Vec3d rvec;
  cv::Mat R_buff2camera_cv;
  cv::eigen2cv(R_buff2camera, R_buff2camera_cv);
  cv::Rodrigues(R_buff2camera_cv, rvec);
  cv::Vec3d tvec(t_buff2camera[0], t_buff2camera[1], t_buff2camera[2]);

  // 重投影预定义模型点
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(OBJECT_POINTS, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}
}  // namespace auto_buff