#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(
  const std::string & config_path, const std::string & can, const std::string & output_folder)
{
  // 读取标定板参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int count = 0;
  while (true) {
    // tools::logger()->info("Start reading camera...");
    camera.read(img, timestamp);
    // tools::logger()->info("Camera read finished, reading IMU...");
    Eigen::Quaterniond q = cboard.imu_at(timestamp);
    // tools::logger()->info("IMU read finished.");
    // --- 新增调试代码开始 ---
    
    // 1. 计算总旋转角度 (Total Rotation Angle)
    // 公式: theta = 2 * arccos(w)
    // 注意：acos 的输入范围是 [-1, 1]，为了安全起见可以 clamp 一下，但在正常四元数下通常不需要
    double angle_rad = 2.0 * std::acos(std::min(std::max(q.w(), -1.0), 1.0));
    double angle_deg = angle_rad * 180.0 / CV_PI; // 转换为角度

    // 2. 设定阈值颜色
    // 如果旋转角度大于 5 度，显示绿色(合格)；否则显示红色(幅度太小)
    // OpenCV 的颜色顺序是 BGR
    cv::Scalar status_color = (std::abs(angle_deg) > 5.0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Yaw(Z) {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Pitch(Y) {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Roll(X) {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    // 4. 显示新增的关键调试信息
    // 显示四元数 w 值 (越接近 1 说明越静止)
    tools::draw_text(img_with_ypr, fmt::format("Q.w: {:.4f}", q.w()), {40, 160}, {255, 255, 255}); // 白色
    
    // 显示总旋转角度 (这是最重要的指标！)
    tools::draw_text(img_with_ypr, fmt::format("Total Angle: {:.2f}", angle_deg), {40, 200}, status_color);

    // 5. 显示操作提示
    std::string hint = (std::abs(angle_deg) > 15.0) ? "GOOD! HOLD & SPACE" : "MOVE MORE!";
    tools::draw_text(img_with_ypr, hint, {40, 250}, status_color);

    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按"s"保存图片和对应四元数，按"q"退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q' || key == 'Q')
      break;
    if (key == 's' || key == 'S') {
      // 保存图片和四元数
      count++;
      auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
      auto q_path = fmt::format("{}/{}.txt", output_folder, count);
      cv::imwrite(img_path, img);
      write_q(q_path, q);
      tools::logger()->info("[{}] Saved in {}", count, output_folder);
    }
  }

  // 离开该作用域时，camera和cboard会自动关闭
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  // 主循环，保存图片和对应四元数
  capture_loop(config_path, "can0", output_folder);

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
