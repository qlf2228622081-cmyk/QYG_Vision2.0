#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder(100);  //根据实际帧率调整

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter, false);

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      camera.read(img, t);
      detector.push(img, t);
    }
  });

  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

  while (!exiter.exit()) {
    auto t0 = std::chrono::steady_clock::now();
    /// 自瞄核心逻辑
    auto [img, armors, t] = detector.debug_pop();
    Eigen::Quaterniond q = cboard.imu_at(t - 1ms);
    mode = cboard.mode;

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", io::MODES[mode]);
      last_mode = mode;
    }

    solver.set_R_gimbal2world(q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
    tracker.set_enemy_color(cboard.enemy_color_string());
    auto targets = tracker.track(armors, t);

    commandgener.push(targets, t, cboard.bullet_speed, ypr);  // 发送给决策线程

    /// debug
    auto fps = 1.0 / tools::delta_time(std::chrono::steady_clock::now(), t0);
    tools::draw_text(img, fmt::format("[{}], FPS: {:.2f}, Mode: {}", tracker.state(), fps, io::MODES[mode]), {10, 30}, {255, 255, 255});
    tools::draw_text(img,fmt::format("Gimbal:Yaw: {:.2f}, Pitch: {:.2f}, Bullet Speed: {:.2f}", ypr[0]*57.3, ypr[1]*57.3, cboard.bullet_speed), {10, 60}, {255, 255, 255});
    tools::draw_text(img,fmt::format("Command: Yaw: {:.2f}, Pitch: {:.2f}", commandgener.yaw*57.3, commandgener.pitch*57.3), {10, 90}, {255, 255, 255});

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      tools::draw_text(img, fmt::format("angle: {:.2f} m", x[6]*57.3), {10, 120}, {255, 255, 255});
      tools::draw_text(img, fmt::format("angle_rate: {:.2f} m", x[7]), {10, 150}, {255, 255, 255});
      tools::draw_text(img, fmt::format("R: {:.2f} m", x[8]), {10, 180}, {255, 255, 255});

    }

    recorder.record(img,q,t);

  }

  detect_thread.join();

  return 0;
}
