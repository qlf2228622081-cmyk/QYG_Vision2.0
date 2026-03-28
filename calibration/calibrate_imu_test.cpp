#include <fmt/core.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <chrono>

// 引入你项目中的现有驱动和工具
#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

// 简化的命令行参数，只需要配置文件路径
const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 1. 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  // 2. 初始化硬件驱动 (完全复用 capture.cpp 的逻辑)
  tools::logger()->info("正在初始化相机与IMU...");
  
  // 初始化 IMU (CBoard 内部应该包含了串口通信和解算)
  io::CBoard cboard(config_path);
  // 初始化 相机
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  tools::logger()->info("初始化完成！按 'q' 或 'ESC' 退出。");
  tools::logger()->info("请晃动云台，检查屏幕上的 YPR 角度变化是否符合预期。");

  while (true) {
    // -----------------------------------------------------------
    // 3. 数据读取 (核心逻辑)
    // -----------------------------------------------------------
    
    // 读取图像和时间戳
    // camera.read 可能会阻塞等待图像，这正好控制了循环频率
    camera.read(img, timestamp);
    
    if (img.empty()) {
        tools::logger()->error("读取图像失败！");
        continue;
    }

    // 根据图像的时间戳，插值获取对应的 IMU 四元数
    Eigen::Quaterniond q = cboard.imu_at(timestamp);

    // -----------------------------------------------------------
    // 4. 数据计算与可视化
    // -----------------------------------------------------------
    
    auto debug_img = img.clone();

    // A. 计算欧拉角 (Z-Y-X 顺序: Yaw, Pitch, Roll)
    // 使用 tools::eulers 保持和你 capture 代码一致的算法
    Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0) * 57.29578; // 弧度转角度

    // B. 计算总旋转角度 (用于检测静止状态)
    // w 越接近 1，说明转动角度越小。 angle = 2 * acos(w)
    double angle_total = 2.0 * std::acos(std::min(std::max(q.w(), -1.0), 1.0)) * 57.29578;

    // C. 绘制信息背景板 (黑色半透明背景，让字看得更清)
    cv::rectangle(debug_img, cv::Rect(20, 20, 300, 220), cv::Scalar(0, 0, 0), -1); 

    // D. 绘制核心数据
    int x = 30;
    int y = 50;
    int dy = 35;
    auto color_val = cv::Scalar(0, 255, 255); // 黄色
    auto color_lbl = cv::Scalar(200, 200, 200); // 灰色

    // 显示 Yaw (Z)
    tools::draw_text(debug_img, fmt::format("Yaw   (Z): {:6.2f}", ypr[0]), {x, y}, color_val);
    // 显示 Pitch (Y)
    tools::draw_text(debug_img, fmt::format("Pitch (Y): {:6.2f}", ypr[1]), {x, y + dy}, color_val);
    // 显示 Roll (X)
    tools::draw_text(debug_img, fmt::format("Roll  (X): {:6.2f}", ypr[2]), {x, y + dy * 2}, color_val);

    // 显示原始四元数 (用于检查数据是否卡死，或者顺序是否反了)
    // w 如果一直不动，或者 w 很小，都需要警惕
    y += dy * 3;
    tools::draw_text(debug_img, fmt::format("Q.w: {:.4f}", q.w()), {x, y}, cv::Scalar(255, 255, 255));
    tools::draw_text(debug_img, fmt::format("Q.x: {:.4f}", q.x()), {x, y + 25}, cv::Scalar(180, 180, 180));
    
    // 绘制中心十字准星 (辅助你看图像中心)
    cv::drawMarker(debug_img, {debug_img.cols / 2, debug_img.rows / 2}, cv::Scalar(0, 255, 0), cv::MARKER_CROSS, 20, 2);

    // -----------------------------------------------------------
    // 5. 显示与控制
    // -----------------------------------------------------------
    cv::resize(debug_img, debug_img, {}, 0.5, 0.5); // 缩小显示，防止屏幕放不下
    cv::imshow("Camera & IMU Inspector", debug_img);

    char key = cv::waitKey(1);
    if (key == 'q' || key == 'Q' || key == 27) { // 27 is ESC
        break;
    }
  }

  return 0;
}