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

  io::CBoard cboard(config_path);//和主控板通信
  io::Camera camera(config_path);//相机读取

<<<<<<< HEAD
  auto_aim::multithread::MultiThreadDetector detector(config_path,true);//装甲板检测
  auto_aim::Solver solver(config_path);//姿态和坐标解算
  auto_aim::Tracker tracker(config_path, solver);//目标跟踪
  auto_aim::Aimer aimer(config_path);//瞄准解算
  auto_aim::Shooter shooter(config_path);//决定是否开火
  io::Command command;

  tools::ThreadSafeQueue<AimTask, true> target_queue(1);
  target_queue.push({std::nullopt, std::chrono::steady_clock::now()});

  // auto_buff::Buff_Detector buff_detector(config_path);
  // auto_buff::Solver buff_solver(config_path);
  // auto_buff::SmallTarget buff_small_target;
  // auto_buff::BigTarget buff_big_target;
  // auto_buff::Aimer buff_aimer(config_path);

  std::atomic<bool> quit = false;

  std::atomic<io::Mode> mode{io::Mode::idle};
  std::atomic<io::Mode> last_mode{io::Mode::idle};
  int idle_counter = 0;  // idle模式下的计数器，用于降低发送频率
  constexpr int send_repeat_count = 10;  // 每个命令重复发送次数，提高数据到达率

  // 检测线程：异步进行图像采集和检测（负责找目标）
=======
  auto_aim::multithread::MultiThreadDetector detector(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter, false);

>>>>>>> wlc
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      camera.read(img, t);
      detector.push(img, t);
    }
  });

<<<<<<< HEAD
  // 瞄准线程：使用Aimer进行瞄准（负责打目标）
  auto plan_thread = std::thread([&]() {
    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto task = target_queue.pop();
        std::list<auto_aim::Target> targets;
        if (task.target.has_value()) {
          targets.push_back(task.target.value());
        }

        auto plan = aimer.aim(targets, task.timestamp, cboard.bullet_speed, true);
        const Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
        plan.shoot = shooter.shoot(plan, aimer, targets, gimbal_pos);

        cboard.send(plan);

        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(10ms);  // 减少等待时间，提高响应速度
      }
    }
  });
  //这是主循环，负责模式切换和发送命令
=======
  auto mode = io::Mode::idle;
  auto last_mode = io::Mode::idle;

>>>>>>> wlc
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

<<<<<<< HEAD
      auto q = cboard.imu_at(t);
      recorder.record(img, q, t); 
      solver.set_R_gimbal2world(q);
      auto targets = tracker.track(armors, t);//把检测结果变成稳定的跟踪目标
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
=======
    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto targets = tracker.track(armors, t);

    commandgener.push(targets, t, cboard.bullet_speed, ypr);  // 发送给决策线程

    /// debug
    auto fps = 1.0 / tools::delta_time(std::chrono::steady_clock::now(), t0);
    tools::draw_text(img, fmt::format("[{}], FPS: {:.2f}, Mode: {}", tracker.state(), fps, io::MODES[mode]), {10, 30}, {255, 255, 255});
    tools::draw_text(img,fmt::format("Gimbal:Yaw: {:.2f}, Pitch: {:.2f}, Bullet Speed: {:.2f}", ypr[0]*57.3, ypr[1]*57.3, cboard.bullet_speed), {10, 60}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "Command: Yaw: {:.2f}, Pitch: {:.2f}", commandgener.yaw.load() * 57.3,
        commandgener.pitch.load() * 57.3),
      {10, 90}, {255, 255, 255});

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
>>>>>>> wlc
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
