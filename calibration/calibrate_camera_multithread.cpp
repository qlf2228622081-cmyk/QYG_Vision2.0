#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <future>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }"
  "{threads t      | 8                        | 并行线程数       }";

struct ImageData
{
  int index;
  cv::Mat img;
  std::vector<cv::Point2f> corners;
  bool success;
};

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;
  
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});
  
  return centers_3d;
}

ImageData process_single_image(
  int index, const std::string & img_path, const cv::Size & pattern_size)
{
  ImageData result;
  result.index = index;
  result.success = false;
  
  try {
    // 读取图片
    result.img = cv::imread(img_path);
    if (result.img.empty()) {
      return result;
    }
    
    // 识别标定板
    auto success = cv::findChessboardCorners(
      result.img, pattern_size, result.corners,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    
    if (success) {
      // 棋盘格检测成功，进行亚像素优化
      cv::Mat gray;
      cv::cvtColor(result.img, gray, cv::COLOR_BGR2GRAY);
      cv::cornerSubPix(
        gray, result.corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
      result.success = true;
    }
    
    fmt::print("[{}] {}: {}\n", index, img_path, success ? "success" : "failure");
    
  } catch (const std::exception & e) {
    fmt::print("[{}] Error processing {}: {}\n", index, img_path, e.what());
  }
  
  return result;
}

void load_parallel(
  const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
  std::vector<std::vector<cv::Point3f>> & obj_points,
  std::vector<std::vector<cv::Point2f>> & img_points,
  int num_threads)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  cv::Size pattern_size(pattern_cols, pattern_rows);
  
  // 首先找出所有需要处理的图片
  std::vector<int> image_indices;
  for (int i = 1; true; i++) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break;
    image_indices.push_back(i);
    if (i == 1) {
      img_size = img.size();
    }
  }
  
  fmt::print("Found {} images, processing with {} threads...\n", image_indices.size(), num_threads);
  
  // 并行处理图像
  std::vector<std::future<ImageData>> futures;
  for (int i : image_indices) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    
    futures.push_back(std::async(
      std::launch::async, process_single_image, i, img_path, pattern_size));
    
    // 限制同时运行的线程数
    if (futures.size() >= static_cast<size_t>(num_threads)) {
      // 等待一个完成
      auto result = futures.front().get();
      futures.erase(futures.begin());
      
      // 显示结果
      if (result.success) {
        auto drawing = result.img.clone();
        cv::drawChessboardCorners(drawing, pattern_size, result.corners, true);
        cv::resize(drawing, drawing, {}, 0.5, 0.5);
        cv::imshow("Press any to continue", drawing);
        cv::waitKey(1);  // 非阻塞显示
        
        img_points.emplace_back(result.corners);
        obj_points.emplace_back(centers_3d(pattern_size, center_distance_mm));
      }
    }
  }
  
  // 处理剩余的任务
  for (auto & future : futures) {
    auto result = future.get();
    
    if (result.success) {
      auto drawing = result.img.clone();
      cv::drawChessboardCorners(drawing, pattern_size, result.corners, true);
      cv::resize(drawing, drawing, {}, 0.5, 0.5);
      cv::imshow("Press any to continue", drawing);
      cv::waitKey(1);
      
      img_points.emplace_back(result.corners);
      obj_points.emplace_back(centers_3d(pattern_size, center_distance_mm));
    }
  }
  
  fmt::print("Successfully processed {} images\n", img_points.size());
}

void print_yaml(const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs, double error)
{
  YAML::Emitter result;
  std::vector<double> camera_matrix_data(
    camera_matrix.begin<double>(), camera_matrix.end<double>());
  std::vector<double> distort_coeffs_data(
    distort_coeffs.begin<double>(), distort_coeffs.end<double>());
  
  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("重投影误差: {:.4f}px", error));
  result << YAML::Key << "camera_matrix";
  result << YAML::Value << YAML::Flow << camera_matrix_data;
  result << YAML::Key << "distort_coeffs";
  result << YAML::Value << YAML::Flow << distort_coeffs_data;
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
    
    if (num_threads < 1) num_threads = 1;
    if (num_threads > std::thread::hardware_concurrency()) {
      num_threads = std::thread::hardware_concurrency();
      fmt::print("Warning: threads number limited to {}\n", num_threads);
    }
    
    // 从输入文件夹中加载标定所需的数据（并行处理）
    cv::Size img_size;
    std::vector<std::vector<cv::Point3f>> obj_points;
    std::vector<std::vector<cv::Point2f>> img_points;
    load_parallel(input_folder, config_path, img_size, obj_points, img_points, num_threads);
    
    if (img_points.empty()) {
      fmt::print("Error: No valid images found for calibration!\n");
      return 1;
    }
    
    fmt::print("Starting camera calibration with {} images...\n", img_points.size());
    
    // 相机标定（OpenCV内部已优化，支持多线程）
    cv::Mat camera_matrix, distort_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    
    auto criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
    
    int flags = cv::CALIB_FIX_K3;
    
    // 使用OpenCV的并行优化
    cv::setNumThreads(num_threads);
    
    cv::calibrateCamera(
      obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs, flags,
      criteria);
    
    // 重投影误差（并行计算）
    std::vector<std::future<std::pair<double, size_t>>> error_futures;
    const size_t batch_size = (obj_points.size() + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
      error_futures.push_back(std::async(std::launch::async, [&, t]() -> std::pair<double, size_t> {
        size_t start = t * batch_size;
        size_t end = std::min(start + batch_size, obj_points.size());
        
        double local_error_sum = 0;
        size_t local_total = 0;
        
        for (size_t i = start; i < end; i++) {
          std::vector<cv::Point2f> reprojected_points;
          cv::projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);
          
          local_total += reprojected_points.size();
          for (size_t j = 0; j < reprojected_points.size(); j++) {
            local_error_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
          }
        }
        
        return {local_error_sum, local_total};
      }));
    }
    
    // 收集结果
    double error_sum = 0;
    size_t total_points = 0;
    for (auto & future : error_futures) {
      auto [local_error, local_total] = future.get();
      error_sum += local_error;
      total_points += local_total;
    }
    
    auto error = error_sum / total_points;
    
    // 输出yaml
    print_yaml(camera_matrix, distort_coeffs, error);
    
  } catch (const std::exception & e) {
    fmt::print("Fatal error: {}\n", e.what());
    return 1;
  }
  
  return 0;
}
