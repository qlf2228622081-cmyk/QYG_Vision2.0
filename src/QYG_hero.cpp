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

/**
 * @brief 英雄机器人(Hero)专用视觉程序
 * 
 * 架构设计:
 * 该版本采用了生产者-消费者 (Producer-Consumer) 模型，通过多线程分离“图像检测”与“瞄准规划”：
 * 1. 检测线程 (Detection Thread): 负责从相机高频取图并进行 YOLO 推理，产出装甲板列表。
 * 2. 瞄准线程 (Planning Thread): 负责从队列获取最新目标，运行弹道补偿，并稳定下发控制指令。
 * 这样可以最大化发挥硬件性能，降低因为推理耗时导致的控制链路抖动。
 */
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

/**
 * @brief 瞄准任务结构体
 * 用于在处理管线间传递目标的运动状态和对应的时间基准。
 */
struct AimTask
{
  std::optional<auto_aim::Target> target; // 跟踪到的目标（含 EKF 预测状态）
  std::chrono::steady_clock::time_point timestamp; // 目标对应的时间戳
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

  // 2. 核心调试与信号管理组件
  tools::Exiter exiter;
  tools::Recorder recorder;

  // 3. 硬件接口初始化
  io::CBoard cboard(config_path); // 初始化电控主控板通信 (CAN/Serial)
  io::Camera camera(config_path); // 初始化工业相机驱动

  // 4. 算法模块初始化
  // MultiThreadDetector 内部维护了一个检测队列和独立的 GPU 推理线程
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);             // 视觉位姿/空间三角化解算
  auto_aim::Tracker tracker(config_path, solver);   // 多目标 EKF (扩展卡尔曼滤波) 跟踪器
  auto_aim::Aimer aimer(config_path);               // 英雄专用弹道补偿回归
  auto_aim::Shooter shooter(config_path);           // 自动打击/射击策略管控
  io::Command command;

  // 5. 跨线程同步队列
  // 将识别出的目标推送到此队列，解耦检测与控制逻辑
  tools::ThreadSafeQueue<AimTask, true> target_queue(1);
  target_queue.push({std::nullopt, std::chrono::steady_clock::now()});

  std::atomic<bool> quit = false;
  std::atomic<io::Mode> mode{io::Mode::idle};
  std::atomic<io::Mode> last_mode{io::Mode::idle};
  int idle_counter = 0;  
  constexpr int send_repeat_count = 10; 

  /**
   * @brief 检测线程 (Detection Thread)
   * 职责: 负责高频率、零延迟地从相机读取原始图像，并压入 detector 异步推理队列。
   * 即使后端处理变慢，该线程也能确保相机内部缓冲区始终处于最新状态。
   */
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t);  // 将图像推入多线程推理管线
      } else
        std::this_thread::sleep_for(10ms);
    }
  });

  /**
   * @brief 瞄准与规划线程 (Planning Thread)
   * 职责: 从同步队列中弹出目标数据，计算弹道补偿并下发指令。
   * 该线程独立于主循环运行，确保由于串口波动或系统调度引起的微量延迟不会中断姿态控制。
   */
  auto plan_thread = std::thread([&]() {
    while (!quit && !exiter.exit()) {
      // 仅在自瞄模式且有目标更新时运行
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto task = target_queue.pop();
        std::list<auto_aim::Target> targets;
        if (task.target.has_value()) {
          targets.push_back(task.target.value());
        }

        // 核心计算: 求解目标在世界坐标系下的预测落点及角度
        auto plan = aimer.aim(targets, task.timestamp, cboard.bullet_speed, true);
        const Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
        
        // 决策判断: 是否满足射击范围限制与弹速对齐要求
        plan.shoot = shooter.shoot(plan, aimer, targets, gimbal_pos);

        // 控制输出: 通过 CAN 总线发送序列化后的控制包
        cboard.send(plan);

        std::this_thread::sleep_for(10ms); // 指令刷新率定为 100Hz
      } else {
        std::this_thread::sleep_for(10ms); 
      }
    }
  });

  /**
   * @brief 主循环 (Main Loop)
   * 职责: 负责感知层的逻辑编排、模式管理以及核心追踪器 (Tracker) 的更新。
   */
  while (!exiter.exit()) {
    mode = cboard.mode; // 监听电控反馈的模式变更
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("英雄视觉核心切换至: {}", io::MODES[current_mode]);
      last_mode.store(current_mode);
    }

    // --- [自瞄任务流程] ---
    if (current_mode == io::Mode::auto_aim) {
      // 获取异步线程处理完成的识别结果
      auto [img, armors, t] = detector.debug_pop();

      // 位姿对齐: 利用时间戳从历史队列检索云台姿态
      auto q = cboard.imu_at(t);
      recorder.record(img, q, t); // 关键帧数据打标记录
      
      solver.set_R_gimbal2world(q);
      
      // 更新 EKF 状态量: 实现目标的运动轨迹平滑与预测
      auto targets = tracker.track(armors, t);
      
      // 更新给瞄准线程的数据包
      if (!targets.empty()) {
        target_queue.push({targets.front(), t});
      } else {
        target_queue.push({std::nullopt, t});
      }
    }

    // --- [空闲/挂机模式] ---
    else if (current_mode == io::Mode::idle) {
      command = {false, false, 0, 0, 0};
      // idle 模式下保持低频心跳发送，维持链路活跃
      if (++idle_counter >= 10) {
        for (int i = 0; i < send_repeat_count; ++i) {
          cboard.send(command);
        }
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);
    }
  }

  // 组件资源回收与停止运行
  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  
  // 退出前强制复位云台位置
  command = {false, false, 0, 0, 0};
  cboard.send(command); 

  return 0;
}