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
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

/**
 * @brief 哨兵机器人 (Sentry) 专用视觉程序
 * 哨兵作为自动防守核心，通常具有双云台或更复杂的自动巡逻算法。
 * 该程序适配了哨兵的自动瞄准逻辑。
 */
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

struct AimTask
{
  std::optional<auto_aim::Target> target;
  std::chrono::steady_clock::time_point timestamp;
};

int main(int argc, char * argv[])
{
  // 1. 初始化命令行解析
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;

  // 2. 硬件与接口初始化
  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  // 3. 算法模块初始化
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  io::Command command;

  // 4. 指令生成同步队列
  tools::ThreadSafeQueue<AimTask, true> target_queue(1);
  target_queue.push({std::nullopt, std::chrono::steady_clock::now()});

  std::atomic<bool> quit = false;
  std::atomic<io::Mode> mode{io::Mode::idle};
  std::atomic<io::Mode> last_mode{io::Mode::idle};
  int idle_counter = 0;  
  constexpr int send_repeat_count = 10; 

  /**
   * @brief 检测线程: 负责图像获取与异步识别
   */
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t); 
      } else
        std::this_thread::sleep_for(10ms);
    }
  });

  /**
   * @brief 决策线程: 根据目标状态生成运动控制指令
   */
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
        std::this_thread::sleep_for(10ms); 
      }
    }
  });

  /**
   * @brief 主业务循环: 更新 EKF 状态与记录元数据
   */
  while (!exiter.exit()) {
    mode = cboard.mode;
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("哨兵视觉模式切换至: {}", io::MODES[current_mode]);
      last_mode.store(current_mode);
    }

    if (current_mode == io::Mode::auto_aim) {
      auto [img, armors, t] = detector.debug_pop();

      auto q = cboard.imu_at(t);
      recorder.record(img, q, t); 
      
      solver.set_R_gimbal2world(q);
      
      // 更新 EKF 预测器
      auto targets = tracker.track(armors, t);
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
      }
    }

    else if (current_mode == io::Mode::idle) {
      command = {false, false, 0, 0, 0};
      if (++idle_counter >= 10) {
        for (int i = 0; i < send_repeat_count; ++i) {
          cboard.send(command);
        }
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);
    }
  }

  // 优雅停止
  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  command = {false, false, 0, 0, 0};
  cboard.send(command); 

  return 0;
}
