#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <fstream>
#include <sstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/logger.hpp"

// 手眼标定过程让标定板不动，改变机械手位置采集图片。Opencv官方提示：至少需要两个具有非平行旋转轴的运动。因此，至少需要 3 个不同的姿势，但强烈建议使用更多的姿势

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

std::vector<cv::Point3f> centers_3d(
  const cv::Size & pattern_size, const float center_distance, const bool is_asymmetric)
{
  std::vector<cv::Point3f> centers_3d;
  centers_3d.reserve(pattern_size.width * pattern_size.height);

  for (int i = 0; i < pattern_size.height; i++) {
    for (int j = 0; j < pattern_size.width; j++) {
      const float x = is_asymmetric ? static_cast<float>((2 * j + (i % 2)) * center_distance)
                                    : static_cast<float>(j * center_distance);
      const float y = static_cast<float>(i * center_distance);
      centers_3d.push_back({x, y, 0.0f});
    }
  }

  return centers_3d;
}

Eigen::Quaterniond read_q(const std::string & q_path)
{
  std::ifstream q_file(q_path);
  double w, x, y, z;
  q_file >> w >> x >> y >> z;
  return {w, x, y, z};
}

void load(
  const std::string & input_folder, const std::string & config_path,
  std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
  std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  auto circle_grid_type = yaml["circle_grid_type"] ? yaml["circle_grid_type"].as<std::string>() : "symmetric";
  bool is_asymmetric_circle_grid = circle_grid_type == "asymmetric";
  R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  cv::SimpleBlobDetector::Params blob_params;
  blob_params.filterByArea = true;
  blob_params.minArea = 16.0f;
  blob_params.maxArea = 1e6f;
  blob_params.filterByCircularity = true;
  blob_params.minCircularity = 0.5f;
  blob_params.filterByConvexity = false;
  blob_params.filterByInertia = false;
  blob_params.filterByColor = false;
  blob_params.minDistBetweenBlobs = 5.0f;
  auto blob_detector = cv::SimpleBlobDetector::create(blob_params);

  const int circles_grid_flags =
    (is_asymmetric_circle_grid ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID) |
    cv::CALIB_CB_CLUSTERING;

  for (int i = 1; true; i++) {
    // 读取图片和对应四元数
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);
    auto img = cv::imread(img_path);
    Eigen::Quaterniond q = read_q(q_path);
    if (img.empty()) break;

    // 计算云台的欧拉角
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =   
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    std::ostringstream oss;
    oss << R_imubody2imuabs;
    tools::logger()->info("R_imubody2imuabs:\n{}\n", oss.str());
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3; // degree

    // 在图片上显示云台的欧拉角，用来检验R_gimbal2imubody是否正确
    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

    // 识别圆点阵列
    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findCirclesGrid(
      img, pattern_size, centers_2d, circles_grid_flags, blob_detector);

    if (success) {
      cv::Mat gray;
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(
        gray, centers_2d, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }

    // 显示识别结果
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("Press any to continue", drawing);
    cv::waitKey(0);

    // 输出识别结果
    fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
    if (!success) continue;

    // 计算所需的数据
    cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat R_gimbal2world_cv;
    cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
    cv::Mat rvec, tvec;
    auto centers_3d_ = centers_3d(pattern_size, center_distance_mm, is_asymmetric_circle_grid);
    cv::solvePnP(
      centers_3d_, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    cv::Mat R;
    cv::Rodrigues(rvec, R);


    // 记录所需的数据
    R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
    t_gimbal2world_list.emplace_back(t_gimbal2world);
    rvecs.emplace_back(R);
    tvecs.emplace_back(tvec);
  }
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  YAML::Emitter result;
  std::vector<double> R_camera2gimbal_data(
    R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_camera2gimbal_data(
    t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  result << YAML::BeginMap;
  result << YAML::Key << "R_gimbal2imubody";
  result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
  "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
  result << YAML::Key << "R_camera2gimbal";
  result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
  result << YAML::Key << "t_camera2gimbal";
  result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
  result << YAML::Newline;
  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  // 从输入文件夹中加载标定所需的数据
  std::vector<double> R_gimbal2imubody_data;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;
  load(
    input_folder, config_path, R_gimbal2imubody_data, R_gimbal2world_list, t_gimbal2world_list,
    rvecs, tvecs);

  // 手眼标定
  cv::Mat R_camera2gimbal, t_camera2gimbal;
    // 在 calibrateHandEye 之前
  std::cout << "------- DEBUG TRACE -------" << std::endl;
  // 取第 5 帧和第 10 帧（假设这期间你有动过）
  int i = 5; 
  int j = 10; 
  if (rvecs.size() > j) {
    // 1. 视觉算出的旋转向量差
    cv::Mat rvec_diff = rvecs[j] - rvecs[i];
    std::cout << "Vision Delta (rvec): " << rvec_diff.t() << std::endl;
    std::cout << "  -> rvec[0] (Pitch/X): " << rvec_diff.at<double>(0) << std::endl;
    std::cout << "  -> rvec[1] (Yaw/Y):   " << rvec_diff.at<double>(1) << std::endl;
    std::cout << "  -> rvec[2] (Roll/Z):  " << rvec_diff.at<double>(2) << std::endl;

      // 2. IMU 算出的旋转差 (欧拉角粗略看)
    Eigen::Matrix3d R_i, R_j;
    cv::cv2eigen(R_gimbal2world_list[i], R_i);
    cv::cv2eigen(R_gimbal2world_list[j], R_j);
    // 计算相对旋转 R_diff = R_i^T * R_j
    Eigen::Matrix3d R_diff_eigen = R_i.transpose() * R_j;
    Eigen::Vector3d euler = R_diff_eigen.eulerAngles(2, 1, 0); // ZYX顺序: Yaw, Pitch, Roll
    std::cout << "IMU Delta (Euler): " << std::endl;
    std::cout << "  -> Pitch (Y): " << euler[1] << " (rad)" << std::endl;
    std::cout << "  -> Yaw (Z):   " << euler[0] << " (rad)" << std::endl;
    std::cout << "  -> Roll (X):  " << euler[2] << " (rad)" << std::endl;
  }
  std::cout << "---------------------------" << std::endl;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // mm to m

  // 计算相机同理想情况的偏角
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  // Eigen::Matrix3d R_gimbal2ideal{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  // Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 2, 1, 0) * 57.3;  // degree

  // 输出yaml
  print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);

  return 0;
}
