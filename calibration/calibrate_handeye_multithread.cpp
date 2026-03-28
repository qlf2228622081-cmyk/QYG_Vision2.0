#include <fmt/core.h>
#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <algorithm>
#include <future>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }"
  "{threads t      | 0                        | 并行线程数(0=自动检测CPU核心数)}";

struct ImageProcessData
{
  int index;
  cv::Mat R_gimbal2world_cv;
  cv::Mat t_gimbal2world;
  cv::Mat rvec;
  cv::Mat tvec;
  bool success;
};

struct CalibrationParams
{
  cv::Size pattern_size;
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody;
  cv::Matx33d camera_matrix;
  cv::Mat distort_coeffs;
  double center_distance_mm;
  std::vector<cv::Point3f> centers_3d;
};

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

Eigen::Quaterniond read_q(const std::string & q_path)
{
  std::ifstream q_file(q_path);
  if (!q_file.is_open()) {
    throw std::runtime_error(
      fmt::format("Cannot open file: {} (file does not exist or permission denied)", q_path));
  }
  
  double w, x, y, z;
  q_file >> w >> x >> y >> z;
  
  if (q_file.fail()) {
    // 尝试读取一行看看文件内容
    q_file.clear();
    q_file.seekg(0);
    std::string line;
    std::getline(q_file, line);
    throw std::runtime_error(
      fmt::format("Invalid quaternion format in: {}. Expected 4 numbers (w x y z), found: '{}'",
                  q_path, line.empty() ? "(empty file)" : line));
  }
  
  // 验证四元数是否归一化（可选，但有助于检测错误）
  double norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (std::abs(norm - 1.0) > 0.1) {
    fmt::print("Warning: Quaternion in {} is not normalized (norm={:.6f}, should be 1.0)\n",
               q_path, norm);
  }
  
  return {w, x, y, z};
}

ImageProcessData process_single_image(
  int index, const std::string & img_path, const std::string & q_path,
  const CalibrationParams & params)
{
  ImageProcessData result;
  result.index = index;
  result.success = false;

  fmt::print("[{}] Processing: {}\n", index, img_path);
  
  try {
    // ========== 阶段1：读取图片 ==========
    fmt::print("[{}] [Step 1] Reading image...\n", index);
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
      fmt::print("[{}] [Step 1] ❌ FAILED: Image file is empty or cannot be read: {}\n", index, img_path);
      return result;
    }
    fmt::print("[{}] [Step 1] ✓ SUCCESS: Image loaded, size: {}x{}\n", index, img.cols, img.rows);

    // ========== 阶段2：读取四元数 ==========
    fmt::print("[{}] [Step 2] Reading quaternion from {}...\n", index, q_path);
    Eigen::Quaterniond q;
    try {
      q = read_q(q_path);
      Eigen::Vector4d xyzw = q.coeffs();
      fmt::print("[{}] [Step 2] ✓ SUCCESS: Quaternion loaded: w={:.6f}, x={:.6f}, y={:.6f}, z={:.6f}\n",
                 index, xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
    } catch (const std::exception & e) {
      fmt::print("[{}] [Step 2] ❌ FAILED: Cannot read quaternion from {}: {}\n", index, q_path, e.what());
      return result;
    }

    // 计算云台的欧拉角
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      params.R_gimbal2imubody.transpose() * R_imubody2imuabs * params.R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;
    fmt::print("[{}] [Step 2] Gimbal pose: Yaw={:.2f}°, Pitch={:.2f}°, Roll={:.2f}°\n",
               index, ypr[0], ypr[1], ypr[2]);

    // ========== 阶段3：识别标定板（棋盘格） ==========
    fmt::print("[{}] [Step 3] Detecting chessboard (pattern size: {}x{})...\n",
               index, params.pattern_size.width, params.pattern_size.height);
    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findChessboardCorners(
        img, params.pattern_size, centers_2d,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!success) {
      fmt::print("[{}] [Step 3] ❌ FAILED: Chessboard not found in image\n", index);
      fmt::print("[{}] [Step 3] Expected {}x{}={} corners, but found {}\n",
                 index, params.pattern_size.width, params.pattern_size.height,
                 params.pattern_size.width * params.pattern_size.height, centers_2d.size());
      return result;
    }
    
    fmt::print("[{}] [Step 3] ✓ SUCCESS: Chessboard detected, found {} corners\n",
               index, centers_2d.size());

    // 棋盘格检测成功，进行亚像素优化
    fmt::print("[{}] [Step 3] Refining corners with sub-pixel accuracy...\n", index);
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::cornerSubPix(
      gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    fmt::print("[{}] [Step 3] ✓ Corner refinement completed\n", index);

    // ========== 阶段4：PnP求解 ==========
    fmt::print("[{}] [Step 4] Solving PnP (camera pose estimation)...\n", index);
    fmt::print("[{}] [Step 4] Camera matrix: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}\n",
               index, params.camera_matrix(0,0), params.camera_matrix(1,1),
               params.camera_matrix(0,2), params.camera_matrix(1,2));
    fmt::print("[{}] [Step 4] 3D points: {}, 2D points: {}\n",
               index, params.centers_3d.size(), centers_2d.size());
    
    cv::Mat rvec, tvec;
    bool pnp_success = false;
    try {
      cv::solvePnP(
        params.centers_3d, centers_2d, params.camera_matrix, params.distort_coeffs, rvec, tvec,
        false, cv::SOLVEPNP_IPPE);
      pnp_success = true;
    } catch (const cv::Exception & e) {
      fmt::print("[{}] [Step 4] ❌ FAILED: PnP solve error: {}\n", index, e.what());
      return result;
    }
    
    if (!pnp_success || rvec.empty() || tvec.empty()) {
      fmt::print("[{}] [Step 4] ❌ FAILED: PnP solve returned empty result\n", index);
      return result;
    }
    
    fmt::print("[{}] [Step 4] ✓ SUCCESS: PnP solved, tvec=[{:.3f}, {:.3f}, {:.3f}]mm\n",
               index, tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    // ========== 所有阶段成功，保存结果 ==========
    fmt::print("[{}] [Final] All steps passed, saving results...\n", index);
    
    // 转换R_gimbal2world到OpenCV格式
    cv::eigen2cv(R_gimbal2world, result.R_gimbal2world_cv);
    result.t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    result.rvec = rvec;
    result.tvec = tvec;
    result.success = true;

    fmt::print("[{}] ✓✓✓ COMPLETE SUCCESS ✓✓✓\n", index);

  } catch (const std::exception & e) {
    fmt::print("[{}] ❌ EXCEPTION: Error processing {}: {}\n", index, img_path, e.what());
    fmt::print("[{}] Exception occurred, marking as failed\n", index);
  } catch (...) {
    fmt::print("[{}] ❌ UNKNOWN EXCEPTION: Unknown error occurred while processing {}\n", index, img_path);
  }

  return result;
}

void load_parallel(
  const std::string & input_folder, const std::string & config_path,
  std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
  std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs, int num_threads)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  // 准备标定参数（所有线程共享）
  CalibrationParams params;
  params.pattern_size = pattern_size;
  params.R_gimbal2imubody = R_gimbal2imubody;
  params.camera_matrix = camera_matrix;
  params.distort_coeffs = distort_coeffs;
  params.center_distance_mm = center_distance_mm;
  params.centers_3d = centers_3d(pattern_size, center_distance_mm);

  // 首先找出所有需要处理的图片
  std::vector<int> image_indices;
  for (int i = 1; true; i++) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;
    image_indices.push_back(i);
  }

  fmt::print("Found {} images, processing with {} threads...\n", image_indices.size(), num_threads);

  // 并行处理图像
  std::vector<std::future<ImageProcessData>> futures;
  std::vector<ImageProcessData> results;  // 用于保持顺序

  for (int i : image_indices) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);

    futures.push_back(std::async(
      std::launch::async, process_single_image, i, img_path, q_path, params));

    // 限制同时运行的线程数
    if (futures.size() >= static_cast<size_t>(num_threads)) {
      // 等待一个完成并收集结果
      auto result = futures.front().get();
      futures.erase(futures.begin());
      results.push_back(result);

      // 显示处理进度（非阻塞）
      if (result.success) {
        // 可以在这里添加进度显示，但不阻塞
        fmt::print("Progress: {}/{}\n", results.size(), image_indices.size());
      }
    }
  }

  // 处理剩余的任务
  for (auto & future : futures) {
    auto result = future.get();
    results.push_back(result);
    if (result.success) {
      fmt::print("Progress: {}/{}\n", results.size(), image_indices.size());
    }
  }

  // 按索引排序结果（保证顺序）
  std::sort(results.begin(), results.end(),
            [](const ImageProcessData & a, const ImageProcessData & b) { return a.index < b.index; });

  // 收集成功的数据
  for (const auto & result : results) {
    if (result.success) {
      R_gimbal2world_list.emplace_back(result.R_gimbal2world_cv);
      t_gimbal2world_list.emplace_back(result.t_gimbal2world);
      rvecs.emplace_back(result.rvec);
      tvecs.emplace_back(result.tvec);
    }
  }

  fmt::print("Successfully processed {} images\n", R_gimbal2world_list.size());
}

void print_yaml(
  const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr_rhr_zyx,
  const Eigen::Vector3d & ypr_ccw_zyx, double theta_deg, const Eigen::Vector3d & axis)
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
    "相机同理想情况的偏角(右手系 ZYX): yaw{:.2f} pitch{:.2f} roll{:.2f} degree",
    ypr_rhr_zyx[0], ypr_rhr_zyx[1], ypr_rhr_zyx[2]));
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角(云台逆时针为正): yaw{:.2f} pitch{:.2f} roll{:.2f} degree",
    ypr_ccw_zyx[0], ypr_ccw_zyx[1], ypr_ccw_zyx[2]));
  result << YAML::Comment(fmt::format(
    "相机同理想情况的角轴: theta={:.2f} deg, axis=[{:.3f}, {:+.3f}, {:+.3f}]",
    theta_deg, axis[0], axis[1], axis[2]));
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
  try {
    // 读取命令行参数
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
      cli.printMessage();
      return 0;
    }
    auto input_folder = cli.get<std::string>(0);
    auto config_path = cli.get<std::string>("config-path");
    auto num_threads = cli.get<int>("threads");

    // 自动检测CPU核心数
    auto max_threads = std::thread::hardware_concurrency();
    if (max_threads == 0) {
      fmt::print("Warning: Cannot detect CPU cores, using 4 threads\n");
      max_threads = 4;
    }

    // 如果用户指定为0或负数，自动使用所有CPU核心
    if (num_threads <= 0) {
      num_threads = max_threads;
      fmt::print("Auto-detected {} CPU cores, using {} threads\n", max_threads, num_threads);
    } else if (num_threads > max_threads) {
      fmt::print(
        "Warning: Requested {} threads, but only {} CPU cores available. Using {} threads\n",
        num_threads, max_threads, max_threads);
      num_threads = max_threads;
    } else {
      fmt::print("Using {} threads (available: {} CPU cores)\n", num_threads, max_threads);
    }

    // 从输入文件夹中加载标定所需的数据（并行处理）
    std::vector<double> R_gimbal2imubody_data;
    std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
    std::vector<cv::Mat> rvecs, tvecs;
    load_parallel(
      input_folder, config_path, R_gimbal2imubody_data, R_gimbal2world_list, t_gimbal2world_list,
      rvecs, tvecs, num_threads);

    if (rvecs.empty()) {
      fmt::print("Error: No valid images found for calibration!\n");
      return 1;
    }

    fmt::print("Starting hand-eye calibration with {} images...\n", rvecs.size());

    // 手眼标定（OpenCV内部已优化）
    cv::Mat R_camera2gimbal, t_camera2gimbal;
    cv::setNumThreads(num_threads);
    cv::calibrateHandEye(
      R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
    t_camera2gimbal /= 1e3;  // mm to m

    // 计算相机同理想情况的偏角（ZYX + 角轴 + 顺时针为正显示）
    Eigen::Matrix3d R_camera2gimbal_eigen;
    cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
    Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
    Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
    Eigen::Vector3d ypr_rhr_zyx = tools::eulers(R_camera2ideal, 2, 1, 0) * 57.3;  // RHR ZYX
    Eigen::Vector3d ypr_ccw_zyx = ypr_rhr_zyx;  // 云台逆时针为正（与右手系一致）

    Eigen::AngleAxisd aa(R_camera2ideal);
    double theta_deg = aa.angle() * 57.29577951308232;  // 180/pi
    Eigen::Vector3d axis = aa.axis();

    fmt::print("[Summary] Camera→Ideal (RHR ZYX): Yaw={:.2f}°, Pitch={:.2f}°, Roll={:.2f}°\n",
           ypr_rhr_zyx[0], ypr_rhr_zyx[1], ypr_rhr_zyx[2]);
    fmt::print("[Summary] Camera→Ideal (CCW positive): Yaw={:.2f}°, Pitch={:.2f}°, Roll={:.2f}°\n",
           ypr_ccw_zyx[0], ypr_ccw_zyx[1], ypr_ccw_zyx[2]);
    fmt::print("[Summary] Camera→Ideal (Angle-Axis): theta={:.2f}°, axis=[{:.3f}, {:+.3f}, {:+.3f}]\n",
           theta_deg, axis[0], axis[1], axis[2]);

    // 输出yaml
    print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal,
           ypr_rhr_zyx, ypr_ccw_zyx, theta_deg, axis);

  } catch (const std::exception & e) {
    fmt::print("Fatal error: {}\n", e.what());
    return 1;
  }

  return 0;
}

