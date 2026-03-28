#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"


const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o|                          | 输出文件夹路径（可选，用于保存可视化图像）}"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }"
  "{max-image-error e| 1.0                    | 每张图允许的平均重投影误差阈值(px)}";

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

void load(
  const std::string & input_folder, const std::string & config_path, cv::Size & img_size,
  std::vector<std::vector<cv::Point3f>> & obj_points,
  std::vector<std::vector<cv::Point2f>> & img_points,
  std::vector<std::string> & img_paths)
{
  // 1. 读取yaml参数 (保持不变)
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  // 2. 循环读取图片 (保持你要求的原始逻辑不变)
  for (int i = 1; true; i++) {
    // 按照 1.jpg, 2.jpg... 顺序读取
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto img = cv::imread(img_path);
    if (img.empty()) break; // 读不到就退出

    // 设置图片尺寸
    img_size = img.size();

    // ==========================================
    // 【修改点 1：精度升级】使用 findChessboardCornersSB
    // ==========================================
    std::vector<cv::Point2f> centers_2d;
    
    // CALIB_CB_EXHAUSTIVE: 穷尽搜索，提高检出率
    // CALIB_CB_ACCURACY: 启用内部高精度亚像素，替代 cornerSubPix
    int flags = cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY;
    
    auto success = cv::findChessboardCornersSB(img, pattern_size, centers_2d, flags);

    // ==========================================
    // 【修改点 2：移除冗余】
    // 原有的 if (success) { cvtColor... cornerSubPix... } 代码块已删除
    // 因为 SB 方法自带了更高精度的亚像素优化，再跑一遍是画蛇添足
    // ==========================================

    // 显示识别结果 (保持不变)
    auto drawing = img.clone();
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 缩小显示
    cv::imshow("Calibration Process", drawing);
    cv::waitKey(100); // 自动播放，100ms一张

    // 输出识别结果
    fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
    if (!success) continue;

    // 记录所需的数据
    img_points.emplace_back(centers_2d);
    obj_points.emplace_back(centers_3d(pattern_size, center_distance_mm));
    img_paths.emplace_back(img_path);
  }
  
  cv::destroyAllWindows();
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

void visualize_calibration_results(
  const std::vector<std::string> & img_paths,
  const std::vector<std::vector<cv::Point3f>> & obj_points,
  const std::vector<std::vector<cv::Point2f>> & img_points,
  const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  const std::vector<cv::Mat> & rvecs, const std::vector<cv::Mat> & tvecs,
  const std::string & output_folder = "")
{
  fmt::print("\n生成标定结果可视化图像...\n");
  
  for (size_t i = 0; i < img_paths.size(); i++) {
    // 读取原始图像
    cv::Mat img = cv::imread(img_paths[i]);
    if (img.empty()) {
      fmt::print(stderr, "警告: 无法读取图像 {}\n", img_paths[i]);
      continue;
    }

    // 计算重投影点
    std::vector<cv::Point2f> reprojected_points;
    cv::projectPoints(
      obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort_coeffs, reprojected_points);

    // 创建可视化图像
    cv::Mat vis_img = img.clone();

    // 绘制检测到的角点（绿色）
    for (const auto & pt : img_points[i]) {
      cv::circle(vis_img, pt, 3, {0, 255, 0}, -1);  // 绿色实心圆
    }

    // 绘制重投影的角点（红色）
    for (const auto & pt : reprojected_points) {
      cv::circle(vis_img, pt, 3, {0, 0, 255}, -1);  // 红色实心圆
    }

    // 绘制误差向量（蓝色线段）
    double max_error = 0.0;
    double error_sum = 0.0;
    for (size_t j = 0; j < img_points[i].size(); j++) {
      cv::Point2f detected = img_points[i][j];
      cv::Point2f reprojected = reprojected_points[j];
      double error = cv::norm(detected - reprojected);
      error_sum += error;
      max_error = std::max(max_error, error);
      
      // 绘制误差向量（放大5倍以便观察）
      cv::Point2f error_vec = (detected - reprojected) * 5.0;
      cv::line(vis_img, reprojected, reprojected + error_vec, {255, 0, 0}, 1);
    }
    double avg_error = error_sum / img_points[i].size();

    // 添加文本信息
    std::string info = fmt::format("Image {}: Avg Error: {:.3f}px, Max Error: {:.3f}px", 
                                    i + 1, avg_error, max_error);
    tools::draw_text(vis_img, info, {10, 30}, {255, 255, 255}, 0.8);
    
    // 添加图例
    cv::circle(vis_img, {10, img.rows - 60}, 3, {0, 255, 0}, -1);
    tools::draw_text(vis_img, "Detected", {20, img.rows - 57}, {0, 255, 0}, 0.6);
    
    cv::circle(vis_img, {10, img.rows - 40}, 3, {0, 0, 255}, -1);
    tools::draw_text(vis_img, "Reprojected", {20, img.rows - 37}, {0, 0, 255}, 0.6);
    
    cv::line(vis_img, {10, img.rows - 20}, {30, img.rows - 20}, {255, 0, 0}, 1);
    tools::draw_text(vis_img, "Error (x5)", {35, img.rows - 17}, {255, 0, 0}, 0.6);

    // 保存或显示图像
    if (!output_folder.empty()) {
      // 提取文件名
      std::string filename = img_paths[i];
      size_t pos = filename.find_last_of("/\\");
      if (pos != std::string::npos) {
        filename = filename.substr(pos + 1);
      }
      // 替换扩展名为 _calib_result.jpg
      pos = filename.find_last_of(".");
      if (pos != std::string::npos) {
        filename = filename.substr(0, pos) + "_calib_result.jpg";
      } else {
        filename += "_calib_result.jpg";
      }
      
      std::string output_path = fmt::format("{}/{}", output_folder, filename);
      cv::imwrite(output_path, vis_img);
      fmt::print("已保存: {}\n", output_path);
    } else {
      // 显示图像
      cv::resize(vis_img, vis_img, {}, 0.5, 0.5);
      cv::imshow(fmt::format("Calibration Result - Image {}", i + 1), vis_img);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
  }
  
  if (!output_folder.empty()) {
    fmt::print("所有可视化图像已保存到: {}\n", output_folder);
  }
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
  cv::Size img_size;
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> img_points;
  std::vector<std::string> img_paths;
  load(input_folder, config_path, img_size, obj_points, img_points, img_paths);

  // 相机标定
  cv::Mat camera_matrix, distort_coeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  
  // 迭代终止条件：棋盘格精度更高，可以使用更严格的精度阈值
  // 棋盘格经过亚像素优化后精度可达0.1像素，圆点阵精度约为0.5像素
  auto criteria = cv::TermCriteria(
    cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100,
    1e-6);  // 使用更严格的精度阈值（从 DBL_EPSILON 改为 1e-6），提高标定精度
  
  // 标定标志：对于棋盘格可以固定主点（如果标定板覆盖足够均匀）
  // 棋盘格检测精度高，可以使用更多约束；圆点阵可以保持灵活性
  int flags = cv::CALIB_FIX_K3;  // 固定k3（视场角较小）
  // 可选优化：对于棋盘格，可以固定主点以加快收敛
  // flags |= cv::CALIB_FIX_PRINCIPAL_POINT;  // 如果标定板分布足够均匀，可以固定主点
  
  cv::calibrateCamera(
    obj_points, img_points, img_size, camera_matrix, distort_coeffs, rvecs, tvecs, flags,
    criteria);

  // 计算每张图片的平均重投影误差并打印
std::vector<double> img_avg_errors;
img_avg_errors.reserve(obj_points.size());
double error_sum = 0.0;
size_t total_points = 0;
for (size_t i = 0; i < obj_points.size(); ++i) {
  std::vector<cv::Point2f> reprojected_points;
  cv::projectPoints(
      obj_points[i], rvecs[i], tvecs[i],
      camera_matrix, distort_coeffs, reprojected_points);

  double local_sum = 0.0;
  for (size_t j = 0; j < reprojected_points.size(); ++j) {
    local_sum += cv::norm(img_points[i][j] - reprojected_points[j]);
  }
  double avg = local_sum / reprojected_points.size();
  img_avg_errors.emplace_back(avg);
  error_sum += local_sum;
  total_points += reprojected_points.size();
  fmt::print("Image {} avg reproj error: {:.3f}px\n", i + 1, avg);
}
double error = error_sum / static_cast<double>(total_points);

// 基于阈值筛除高误差图片并进行二次标定
const double max_img_error = cli.get<double>("max-image-error");
std::vector<std::vector<cv::Point3f>> obj_points_f;
std::vector<std::vector<cv::Point2f>> img_points_f;
std::vector<std::string> img_paths_f;
obj_points_f.reserve(obj_points.size());
img_points_f.reserve(img_points.size());
img_paths_f.reserve(img_paths.size());

size_t removed = 0;
for (size_t i = 0; i < img_avg_errors.size(); ++i) {
  if (img_avg_errors[i] <= max_img_error) {
    obj_points_f.emplace_back(obj_points[i]);
    img_points_f.emplace_back(img_points[i]);
    img_paths_f.emplace_back(img_paths[i]);
  } else {
    ++removed;
    fmt::print("Filtered out image {} (avg {:.3f}px > {:.3f}px)\n",
               i + 1, img_avg_errors[i], max_img_error);
  }
}

if (removed > 0 && obj_points_f.size() >= 3) {
  fmt::print("Recalibrating with {} images after filtering (removed {})...\n",
             img_points_f.size(), removed);

  std::vector<cv::Mat> rvecs2, tvecs2;
  cv::calibrateCamera(
      obj_points_f, img_points_f, img_size,
      camera_matrix, distort_coeffs, rvecs2, tvecs2, flags, criteria);

  double err_sum2 = 0.0;
  size_t total2 = 0;
  for (size_t i = 0; i < obj_points_f.size(); ++i) {
    std::vector<cv::Point2f> rp;
    cv::projectPoints(
        obj_points_f[i], rvecs2[i], tvecs2[i],
        camera_matrix, distort_coeffs, rp);
    total2 += rp.size();
    for (size_t j = 0; j < rp.size(); ++j) {
      err_sum2 += cv::norm(img_points_f[i][j] - rp[j]);
    }
  }
  error = err_sum2 / static_cast<double>(total2);

  print_yaml(camera_matrix, distort_coeffs, error);

  std::string output_folder = cli.has("output-folder") ?
      cli.get<std::string>("output-folder") : "";
  visualize_calibration_results(
      img_paths_f, obj_points_f, img_points_f,
      camera_matrix, distort_coeffs, rvecs2, tvecs2, output_folder);
  return 0;
}

// 未筛除或筛除后数量不足，输出首次标定结果并可视化全部
print_yaml(camera_matrix, distort_coeffs, error);
std::string output_folder = cli.has("output-folder") ?
    cli.get<std::string>("output-folder") : "";
visualize_calibration_results(
    img_paths, obj_points, img_points,
    camera_matrix, distort_coeffs, rvecs, tvecs, output_folder);
}