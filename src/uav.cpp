#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

/**
 * @brief 无人机 (UAV) 专用视觉程序
 * 无人机视觉的特点是视角通常从高空俯视，且由于机身晃动剧烈，对 IMU 姿态补偿和目标预测的稳定性要求更高。
 */
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/uav.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 1. 初始化命令行解析
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 2. 调试组件与硬件初始化
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::CBoard cboard(config_path); // 无人机数传接口
  io::Camera camera(config_path); // 无人机下挂相机

  // 3. 算法模块初始化
  auto_aim::YOLO detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  /**
   * @brief 无人机视觉主闭环
   */
  while (!exiter.exit()) {
    // a. 图像采集
    if (!camera.read(img, t)) {
      continue;
    }
    
    // b. 姿态同步 (Drone IMU Data)
    q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;

    if (last_mode != mode) {
      tools::logger()->info("无人机视觉切换为: {}", io::MODES[mode]);
      last_mode = mode;
    }

    // c. 更新世界坐标映射
    solver.set_R_gimbal2world(q);

    // d. 识别环节
    auto armors = detector.detect(img);

    // e. 跟踪与滤波 (EKF 处理空中剧烈抖动)
    auto targets = tracker.track(armors, t);

    // f. 弹道与提前量解算
    auto command = aimer.aim(targets, t, cboard.bullet_speed);

    // g. 发送控制指令 (至无人机飞控或云台)
    cboard.send(command);
  }

  return 0;
}