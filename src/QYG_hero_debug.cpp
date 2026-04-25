#include <chrono>
#include <fmt/core.h>
#include <functional>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/armor.hpp"
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
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

/**
 * @brief 英雄机器人调试程序 (Hero Debug Entry)
 * 相比于标准英雄程序，调试程序增加了：
 * 1. 实时图像显示 (基于 OpenCV `imshow`)。
 * 2. 识别结果的可视化绘制（装甲板角点、编号、置信度等）。
 * 3. 目标预测落点（红框）与当前追踪点（绿框）的对比显示。
 * 4. 指令插值逻辑 (Advanced Command Interpolation)，用于平滑控制频率。
 */
const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明}"
    "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
  // 1. 设置命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 2. 初始化各种调试辅助组件
  tools::Exiter exiter;
  tools::Recorder recorder; // 用于录制数据回放
  tools::Plotter plotter;   // 用于绘制曲线（如预测误差、弹道偏移等）

  // 3. 硬件与接口初始化
  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  // 4. 算法管线初始化
  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  // 线程间数据传递队列
  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  // 用于在多线程间对齐的时间戳原子变量
  std::atomic<std::chrono::steady_clock::time_point> target_timestamp{
      std::chrono::steady_clock::now()};

  std::atomic<bool> quit = false;
  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode{io::Mode::idle};
  int idle_counter = 0;                 
  constexpr int send_repeat_count = 10; 

  // --- 调试模式统计变量 ---
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  double current_fps = 0.0;
  cv::Mat debug_img; // 用于可视化的画布

  /**
   * @brief 检测线程
   * 持续从相机读取图像并压入异步检测流水线。
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
   * @brief 规划线程 (带插值算法)
   * 职责: 根据检测结果计算瞄准角度。
   * 特色逻辑: 指令插值 (Interpolation)。
   * 如果视觉由于光线或遮挡导致断帧，插值逻辑会根据历史命令序列自动补齐中间的微小指令，
   * 避免电控端因为接收频率突降导致云台抖动。
   */
  auto plan_thread = std::thread([&]() {
    // 指令历史缓冲区：记录前三帧的命令和时间点
    std::optional<io::Command> cmd_n_minus_2, cmd_n_minus_1, cmd_n;
    std::optional<std::chrono::steady_clock::time_point> t_n_minus_2, t_n_minus_1, t_n;
    constexpr int interpolation_levels = 3; // 插值深度

    /**
     * @brief 递归线性插值函数
     * 在两个命令点之间按照 level 次级递归内插，使控制指令下发极其平滑。
     */
    std::function<void(const io::Command &, const io::Command &,
                       const std::chrono::steady_clock::time_point &,
                       const std::chrono::steady_clock::time_point &, int)>
        interpolate_and_send;
    interpolate_and_send = [&](const io::Command &cmd1, const io::Command &cmd2,
                               const std::chrono::steady_clock::time_point &t1,
                               const std::chrono::steady_clock::time_point &t2,
                               int level) {
      if (level <= 0) {
        cboard.send(cmd2); // 基础情况：直接发送
        return;
      }
      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
      auto interpolated_time = t2 + dt / 2;

      // 线性内插 Yaw, Pitch 和 测距值
      io::Command interpolated_cmd{
          cmd2.control, 
          cmd2.shoot,
          (cmd1.yaw + cmd2.yaw) / 2.0,
          (cmd1.pitch + cmd2.pitch) / 2.0,
          (cmd1.horizon_distance + cmd2.horizon_distance) / 2.0
      };

      interpolate_and_send(cmd1, interpolated_cmd, t1, interpolated_time, level - 1);
      interpolate_and_send(interpolated_cmd, cmd2, interpolated_time, t2, level - 1);
    };

    while (!quit) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto target = target_queue.pop(); 
        auto timestamp = target_timestamp.load();

        if (target.has_value()) {
          std::list<auto_aim::Target> targets = {target.value()};

          // 核心瞄准角度计算
          auto command = aimer.aim(targets, timestamp, cboard.bullet_speed, true);

          // 计算预测的目标水平距离，用于引导电控端更好的处理弹道
          Eigen::VectorXd x = target->ekf_x();
          double horizon_distance = std::sqrt(x[0] * x[0] + x[2] * x[2]);
          command.horizon_distance = horizon_distance;

          // 综合距离、角度误差判断是否允许自动射击
          Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
          command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

          // 移位更新历史命令队列
          cmd_n_minus_2 = cmd_n_minus_1; cmd_n_minus_1 = cmd_n; cmd_n = command;
          auto now = std::chrono::steady_clock::now();
          t_n_minus_2 = t_n_minus_1; t_n_minus_1 = t_n; t_n = now;
        } else {
          // 无目标时重置指令
          io::Command stop_command{false, false, 0, 0, 0};
          cmd_n_minus_2 = cmd_n_minus_1; cmd_n_minus_1 = cmd_n; cmd_n = stop_command;
          auto now = std::chrono::steady_clock::now();
          t_n_minus_2 = t_n_minus_1; t_n_minus_1 = t_n; t_n = now;
        }
      }

      // 执行插值发送逻辑
      if (cmd_n_minus_1.has_value() && cmd_n_minus_2.has_value() &&
          t_n_minus_1.has_value() && t_n_minus_2.has_value()) {
        interpolate_and_send(cmd_n_minus_2.value(), cmd_n_minus_1.value(),
                             t_n_minus_2.value(), t_n_minus_1.value(),
                             interpolation_levels);
      } else {
        io::Command command_to_send = cmd_n.value_or(
            cmd_n_minus_1.value_or(io::Command{false, false, 0, 0, 0}));
        cboard.send(command_to_send);
      }

      std::this_thread::sleep_for(10ms);
    }
  });

  /**
   * @brief 主循环 (含 UI 绘制逻辑)
   */
  while (!exiter.exit()) {
    mode = cboard.mode;
    nlohmann::json data;
    auto current_mode = mode.load(); 

    if (last_mode != current_mode) {
      tools::logger()->info("英雄调试模式切换为: {}", io::MODES[current_mode]);
      last_mode = current_mode;
    }

    if (current_mode == io::Mode::auto_aim) {
      // 获取识别结果
      auto [img, armors, t] = detector.debug_pop();
      auto q = cboard.imu_at(t - 1ms);

      solver.set_R_gimbal2world(q);
      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      // 为调试曲线添加云台数据
      data["gimbal_yaw"] = ypr[0] * 57.3;
      data["gimbal_pitch"] = ypr[1] * 57.3;

      tracker.set_enemy_color(cboard.enemy_color_string());
      auto targets = tracker.track(armors, t);
      target_timestamp = t; 
      
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      // --- [可视化 UI 绘制] ---
      debug_img = img; 

      // 1. 计算系统 FPS (包含检测与显示的整体耗时)
      auto now = std::chrono::steady_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
      fps_frame_count++;
      if (dt >= 1000) {
        current_fps = fps_frame_count * 1000.0 / dt;
        fps_frame_count = 0;
        last_fps_time = now;
      }

      // 2. 绘制所有检测到的装甲板候选 (绿色点)
      for (const auto &armor : armors) {
        if (!armor.points.empty()) {
          tools::draw_points(debug_img, armor.points, {0, 255, 0}, 2);
        }
        std::string armor_info = fmt::format("{:.2f} {}", armor.confidence, auto_aim::ARMOR_NAMES[armor.name]);
        tools::draw_text(debug_img, armor_info, armor.center, {0, 255, 0}, 0.5, 1);
      }

      // 3. 绘制预测信息
      if (!targets.empty()) {
        auto target = targets.front();

        // 绘制当前正在追踪的目标 (绿框)
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d &xyza : armor_xyza_list) {
          auto image_points = solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_img, image_points, {0, 255, 0}, 2);
        }

        // 绘制 Aimer 预测出的最终击打点 (红框)
        auto aim_point = aimer.debug_aim_point;
        if (aim_point.valid) {
          Eigen::Vector4d aim_xyza = aim_point.xyza;
          auto aim_points = solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_img, aim_points, {0, 0, 255}, 2);
        }
      } else {
        tools::draw_text(debug_img, "No Target Found", {10, 150}, {128, 128, 128}, 0.6, 2);
      }

      // 4. 显示文本信息: FPS、云台角度
      int y_offset = 30;
      std::string fps_text = fmt::format("System FPS: {:.1f}", current_fps);
      tools::draw_text(debug_img, fps_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);

      y_offset += 30;
      Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      std::string gimbal_text = fmt::format("Gimbal Yaw: {:.1f} Pitch: {:.1f}",
                                            gimbal_pos[0] * 57.3, -gimbal_pos[1] * 57.3);
      tools::draw_text(debug_img, gimbal_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);

      // --- [弹出显示窗口] ---
      cv::Mat display_img;
      cv::resize(debug_img, display_img, {}, 0.5, 0.5); // 缩小显示，降低拷贝耗时
      cv::imshow("QYG Hero Debug Terminal", display_img);
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) break;
    }

    else if (current_mode == io::Mode::idle) {
      if (++idle_counter >= 10) {
        io::Command command{false, false, 0, 0, 0};
        for (int i = 0; i < send_repeat_count; ++i) cboard.send(command);
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);
    }
    
    // 发送数据到 Plotter 绘图器
    plotter.plot(data);
  }

  // 资源清理
  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  
  io::Command command{false, false, 0, 0, 0};
  for (int i = 0; i < send_repeat_count; i++) cboard.send(command);

  return 0;
}