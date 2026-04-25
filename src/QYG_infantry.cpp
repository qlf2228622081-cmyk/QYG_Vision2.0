#include <chrono>
#include <functional>
#include <opencv2/opencv.hpp>
#include <optional>
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

/**
 * @brief 步兵机器人 (Infantry) 专用视觉程序
 * 步兵通常搭载 17mm 弹丸，其初速较高且弹道相对英雄更平直。
 * 本程序同样采用生产者-消费者多线程架构，优化控制链路的响应速度。
 */
const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明}"
    "{@config-path   | configs/QYG_infantry.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

/**
 * @brief 瞄准任务同步结构
 */
struct AimTask {
  std::optional<auto_aim::Target> target;
  std::chrono::steady_clock::time_point timestamp;
};

int main(int argc, char *argv[]) {
  // 1. 初始化命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 2. 调试与生命周期工具
  tools::Exiter exiter;
  tools::Recorder recorder;

  // 3. 硬件接口
  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  // 4. 视觉管线组件
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);
  io::Command command;

  // 5. 线程通信队列 (深度为1，仅处理最新帧)
  tools::ThreadSafeQueue<AimTask, true> target_queue(1);
  target_queue.push({std::nullopt, std::chrono::steady_clock::now()});

  std::atomic<bool> quit = false;
  std::atomic<io::Mode> mode{io::Mode::idle};
  std::atomic<io::Mode> last_mode{io::Mode::idle};
  int idle_counter = 0;                 
  constexpr int send_repeat_count = 10; 

  /**
   * @brief 提取线程: 持续从相机读取图像并压入 detector 异步管线
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
   * @brief 瞄准与下发线程: 从队列取出预测目标并发送控制指令
   */
  auto plan_thread = std::thread([&]() {
    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto task = target_queue.pop();
        std::list<auto_aim::Target> targets;
        if (task.target.has_value()) {
          targets.push_back(task.target.value());
        }

        // 调用 Aimer 计算瞄准角度与弹道补偿
        auto plan = aimer.aim(targets, task.timestamp, cboard.bullet_speed, true);
        const Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
        
        // 自动射击策略判断
        plan.shoot = shooter.shoot(plan, aimer, targets, gimbal_pos);

        cboard.send(plan); // 下发最终控制包

        std::this_thread::sleep_for(10ms); // 100Hz 刷新率
      } else {
        std::this_thread::sleep_for(10ms); 
      }
    }
  });

  /**
   * @brief 主循环: 模式同步、图像处理完成后结果回收与 Tracker 更新
   */
  while (!exiter.exit()) {
    mode = cboard.mode;
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("步兵视觉切换至模式: {}", io::MODES[current_mode]);
      last_mode.store(current_mode);
    }

    // --- 自瞄业务逻辑 ---
    if (current_mode == io::Mode::auto_aim) {
      auto [img, armors, t] = detector.debug_pop(); // 获取异步处理完的识别帧

      auto q = cboard.imu_at(t);
      recorder.record(img, q, t);
      
      solver.set_R_gimbal2world(q);
      tracker.set_enemy_color(cboard.enemy_color_string());
      
      // 更新全场目标 EKF 状态
      auto targets = tracker.track(armors, t);
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
      }
    }

    // --- 空闲待机逻辑 ---
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

  // 组件退出处理
  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  
  command = {false, false, 0, 0, 0};
  cboard.send(command); 

  return 0;
}