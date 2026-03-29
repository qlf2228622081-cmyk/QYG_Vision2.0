#include <chrono>
#include <functional>
#include <optional>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
// #include "tasks/auto_buff/buff_aimer.hpp"
// #include "tasks/auto_buff/buff_detector.hpp"
// #include "tasks/auto_buff/buff_solver.hpp"
// #include "tasks/auto_buff/buff_target.hpp"
// #include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

struct AimTask
{
  std::optional<auto_aim::Target> target;
  std::chrono::steady_clock::time_point timestamp;
};

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;

  io::CBoard cboard(config_path);//和主控板通信
  io::Camera camera(config_path);//相机读取

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
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t);  // 异步检测
      } else
        std::this_thread::sleep_for(10ms);
    }
  });

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
  while (!exiter.exit()) {
    mode = cboard.mode;
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", io::MODES[current_mode]);
      last_mode.store(current_mode);
    }

    /// 自瞄
    if (current_mode == io::Mode::auto_aim) {
      // 从检测队列获取结果（异步检测已完成）
      auto [img, armors, t] = detector.debug_pop();

      auto q = cboard.imu_at(t);
      recorder.record(img, q, t); 
      solver.set_R_gimbal2world(q);
      auto targets = tracker.track(armors, t);//把检测结果变成稳定的跟踪目标
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
      }
    }

    // /// 打符
    // else if (mode.load() == io::Mode::small_buff || mode.load() == io::Mode::big_buff) {
    //   buff_solver.set_R_gimbal2world(q);

    //   auto power_runes = buff_detector.detect(img);

    //   buff_solver.solve(power_runes);

    //   auto_aim::Plan buff_plan;
    //   if (mode.load() == io::Mode::small_buff) {
    //     buff_small_target.get_target(power_runes, t);
    //     auto target_copy = buff_small_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   } else if (mode.load() == io::Mode::big_buff) {
    //     buff_big_target.get_target(power_runes, t);
    //     auto target_copy = buff_big_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   }
    //   cboard.send(io::command(plan.control, plan.fire,plan.yaw,plan.pitch,plan.horizon_distance));
    // }
    
    else if (current_mode == io::Mode::idle) {
      command = {false, false, 0, 0, 0};
      // idle模式下降低发送频率（每10次循环发送一次，约500ms）
      if (++idle_counter >= 10) {
        // 发送多次以提高数据到达率（中间可能有损耗）
        for (int i = 0; i < send_repeat_count; ++i) {
          cboard.send(command);
        }
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);  // 降低CPU占用
    }

  }
  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  command = {false, false, 0, 0, 0};
  cboard.send(command);  // 退出前发送停止命令

  return 0;
}