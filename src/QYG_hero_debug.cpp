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

const std::string keys =
    "{help h usage ? |      | 输出命令行参数说明}"
    "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;
  tools::Plotter plotter;

  io::CBoard cboard(config_path);
  io::Camera camera(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  // 用于保存目标时间戳
  std::atomic<std::chrono::steady_clock::time_point> target_timestamp{
      std::chrono::steady_clock::now()};

  // auto_buff::Buff_Detector buff_detector(config_path);
  // auto_buff::Solver buff_solver(config_path);
  // auto_buff::SmallTarget buff_small_target;
  // auto_buff::BigTarget buff_big_target;
  // auto_buff::Aimer buff_aimer(config_path);

  std::atomic<bool> quit = false;

  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode{io::Mode::idle};
  int idle_counter = 0;                 // idle模式下的计数器，用于降低发送频率
  constexpr int send_repeat_count = 10; // 每个命令重复发送次数，提高数据到达率

  // Debug模式相关变量
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  double current_fps = 0.0;
  cv::Mat debug_img; // 用于显示的图像

  // 检测线程：异步进行图像采集和检测
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t); // 异步检测
      } else
        std::this_thread::sleep_for(10ms);
    }
  });

  // 瞄准线程：使用Aimer进行瞄准
  auto plan_thread = std::thread([&]() {
    // 保存历史命令用于插值：cmd(n-2), cmd(n-1), cmd(n)
    std::optional<io::Command> cmd_n_minus_2;
    std::optional<io::Command> cmd_n_minus_1;
    std::optional<io::Command> cmd_n;
    std::optional<std::chrono::steady_clock::time_point> t_n_minus_2;
    std::optional<std::chrono::steady_clock::time_point> t_n_minus_1;
    std::optional<std::chrono::steady_clock::time_point> t_n;
    constexpr int interpolation_levels = 3; // 插值3次，数据翻8倍

    // 递归插值函数：在 cmd1 和 cmd2 之间进行 level 次插值
    // 按照用户公式：t(n-1) + (t(n-1) - t(n-2))/2 计算插值时间点
    std::function<void(const io::Command &, const io::Command &,
                       const std::chrono::steady_clock::time_point &,
                       const std::chrono::steady_clock::time_point &, int)>
        interpolate_and_send;
    interpolate_and_send = [&](const io::Command &cmd1, const io::Command &cmd2,
                               const std::chrono::steady_clock::time_point &t1,
                               const std::chrono::steady_clock::time_point &t2,
                               int level) {
      if (level <= 0) {
        // 基础情况：直接发送 cmd2
        cboard.send(cmd2);
        return;
      }

      // 计算插值时间点：t(n-1) + (t(n-1) - t(n-2))/2
      // 这里 t2 对应 t(n-1)，t1 对应 t(n-2)
      auto dt = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
      auto interpolated_time = t2 + dt / 2; // t(n-1) + (t(n-1) - t(n-2))/2

      // 线性插值命令：在 cmd1 和 cmd2 之间插值
      io::Command interpolated_cmd{
          cmd2.control, // control 和 shoot 使用较新的值
          cmd2.shoot,
          (cmd1.yaw + cmd2.yaw) / 2.0,     // yaw 插值
          (cmd1.pitch + cmd2.pitch) / 2.0, // pitch 插值
          (cmd1.horizon_distance + cmd2.horizon_distance) /
              2.0 // horizon_distance 插值
      };

      // 先发送 cmd1（如果 level > 1，会递归插值）
      interpolate_and_send(cmd1, interpolated_cmd, t1, interpolated_time,
                           level - 1);
      // 然后发送插值后的命令（如果 level > 1，会递归插值）
      interpolate_and_send(interpolated_cmd, cmd2, interpolated_time, t2,
                           level - 1);
    };

    while (!quit) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto target =
            target_queue
                .pop(); // pop()会阻塞等待，但empty()检查可以避免在非自瞄模式下等待
        auto timestamp = target_timestamp.load();

        if (target.has_value()) {
          // 将单个target转换为list
          std::list<auto_aim::Target> targets = {target.value()};

          // 使用Aimer进行瞄准
          auto command =
              aimer.aim(targets, timestamp, cboard.bullet_speed, true);

          // 计算horizon_distance
          Eigen::VectorXd x = target->ekf_x();
          double horizon_distance = std::sqrt(x[0] * x[0] + x[2] * x[2]);
          command.horizon_distance = horizon_distance;

          // 使用Shooter决定是否射击
          Eigen::Vector3d gimbal_pos =
              tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
          command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

          // 更新历史命令
          cmd_n_minus_2 = cmd_n_minus_1;
          cmd_n_minus_1 = cmd_n;
          cmd_n = command;

          tools::logger()->info("Sent Command - Control: {}, Fire: {}, Yaw: "
                                "{:.2f}, Pitch: {:.2f}",
                                command.control, command.shoot,
                                command.yaw * 57.3, command.pitch * 57.3);

          // 更新时间戳
          auto now = std::chrono::steady_clock::now();
          t_n_minus_2 = t_n_minus_1;
          t_n_minus_1 = t_n;
          t_n = now;
        } else {
          // 没有目标，发送停止命令
          io::Command stop_command{false, false, 0, 0, 0};
          cmd_n_minus_2 = cmd_n_minus_1;
          cmd_n_minus_1 = cmd_n;
          cmd_n = stop_command;

          auto now = std::chrono::steady_clock::now();
          t_n_minus_2 = t_n_minus_1;
          t_n_minus_1 = t_n;
          t_n = now;
        }
      }

      // 如果有足够的历史数据（n-1 和 n-2），进行插值发送
      if (cmd_n_minus_1.has_value() && cmd_n_minus_2.has_value() &&
          t_n_minus_1.has_value() && t_n_minus_2.has_value()) {
        // 使用递归插值，操作3次数据翻8倍
        interpolate_and_send(cmd_n_minus_2.value(), cmd_n_minus_1.value(),
                             t_n_minus_2.value(), t_n_minus_1.value(),
                             interpolation_levels);
      } else {
        // 没有足够历史数据，发送当前命令或停止命令
        io::Command command_to_send = cmd_n.value_or(
            cmd_n_minus_1.value_or(io::Command{false, false, 0, 0, 0}));
        cboard.send(command_to_send);
      }

      std::this_thread::sleep_for(10ms);
    }
  });

  while (!exiter.exit()) {
    mode = cboard.mode;
    nlohmann::json data;
    auto current_mode = mode.load(); // 缓存模式值，避免重复调用load()

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", io::MODES[current_mode]);
      last_mode = current_mode;
    }

    /// 自瞄
    if (current_mode == io::Mode::auto_aim) {
      // 从检测队列获取结果（异步检测已完成）
      auto [img, armors, t] = detector.debug_pop();
      auto q = cboard.imu_at(t - 1ms);

      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      data["gimbal_yaw"] = ypr[0] * 57.3;
      data["gimbal_pitch"] = ypr[1] * 57.3;

      // recorder.record(img, q, t);
      solver.set_R_gimbal2world(q);
      tracker.set_enemy_color(cboard.enemy_color_string());
      auto targets = tracker.track(armors, t);
      target_timestamp = t; // 保存时间戳
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      // Debug模式：绘制识别画面和信息
      debug_img = img; // 这里不需要再次 clone，因为后续绘制操作不会影响
                       // recorder 中的图像

      // 计算FPS
      auto now = std::chrono::steady_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_fps_time)
                    .count();
      fps_frame_count++;
      if (dt >= 1000) { // 每秒更新一次FPS
        current_fps = fps_frame_count * 1000.0 / dt;
        fps_frame_count = 0;
        last_fps_time = now;
      }

      // 绘制检测到的装甲板
      for (const auto &armor : armors) {
        // 绘制装甲板四个角点
        if (!armor.points.empty()) {
          tools::draw_points(debug_img, armor.points, {0, 255, 0}, 2);
        }

        // 绘制装甲板信息
        std::string armor_info = fmt::format(
            "{:.2f} {} {} {}", armor.confidence, auto_aim::COLORS[armor.color],
            auto_aim::ARMOR_NAMES[armor.name],
            auto_aim::ARMOR_TYPES[armor.type]);
        tools::draw_text(debug_img, armor_info, armor.center, {0, 255, 0}, 0.5,
                         1);
      }

      // 绘制目标跟踪信息
      if (!targets.empty()) {
        auto target = targets.front();

        // 绘制目标的所有装甲板位置（绿色）
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d &xyza : armor_xyza_list) {
          auto image_points = solver.reproject_armor(
              xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_img, image_points, {0, 255, 0}, 2);
        }

        // 绘制瞄准点（红色）
        auto aim_point = aimer.debug_aim_point;
        if (aim_point.valid) {
          Eigen::Vector4d aim_xyza = aim_point.xyza;
          auto aim_points = solver.reproject_armor(
              aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_img, aim_points, {0, 0, 255}, 2);
        }

        // 显示目标信息
        // Eigen::VectorXd x = target.ekf_x();
        // double distance = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
        // std::string target_info = fmt::format(
        //   "Target: {} | Dist: {:.2f}m | Yaw: {:.1f}deg | W: {:.2f}rad/s",
        //   auto_aim::ARMOR_NAMES[target.name], distance, x[6] * 57.3, x[7]);
        // tools::draw_text(debug_img, target_info, {10, 150}, {255, 255, 0},
        // 0.6, 2);
      } else {
        // 没有目标时显示提示
        tools::draw_text(debug_img, "No Target", {10, 150}, {128, 128, 128},
                         0.6, 2);
      }

      // 显示模式、FPS和装甲板数量
      int y_offset = 30;
      // std::string mode_text = fmt::format("Mode: {}",
      // io::MODES[current_mode]); tools::draw_text(debug_img, mode_text, {10,
      // y_offset}, {255, 255, 255}, 0.7, 2);

      // y_offset += 30;
      // std::string tracker_state_text = fmt::format("Tracker: {}",
      // tracker.state()); tools::draw_text(debug_img, tracker_state_text, {10,
      // y_offset}, {255, 255, 255}, 0.6, 1);

      y_offset += 25;
      std::string fps_text = fmt::format("FPS: {:.1f}", current_fps);
      tools::draw_text(debug_img, fps_text, {10, y_offset}, {255, 255, 255},
                       0.7, 2);

      // y_offset += 30;
      // std::string armor_count_text = fmt::format("Armors: {}",
      // armors.size()); tools::draw_text(debug_img, armor_count_text, {10,
      // y_offset}, {255, 255, 255}, 0.7, 2);

      // 显示云台和命令信息
      y_offset += 30;
      Eigen::Vector3d gimbal_pos =
          tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      std::string gimbal_text =
          fmt::format("Gimbal: Yaw {:.1f}deg Pitch {:.1f}deg",
                      gimbal_pos[0] * 57.3, -gimbal_pos[1] * 57.3);
      tools::draw_text(debug_img, gimbal_text, {10, y_offset}, {255, 255, 255},
                       0.6, 1);

      // y_offset += 25;
      // std::string bullet_text = fmt::format("Bullet Speed: {:.1f} m/s",
      // cboard.bullet_speed); tools::draw_text(debug_img, bullet_text, {10,
      // y_offset}, {255, 255, 255}, 0.6, 1);

      // 显示图像（缩小尺寸以提高性能）
      cv::Mat display_img;
      cv::resize(debug_img, display_img, {}, 0.5, 0.5);
      cv::imshow("QYG Hero Debug", display_img);
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) { // 'q' 或 ESC 退出
        break;
      }
    }

    // /// 打符
    // else if (mode.load() == io::Mode::small_buff || mode.load() ==
    // io::Mode::big_buff) {
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
    //   cboard.send(io::command(plan.control,
    //   plan.fire,plan.yaw,plan.pitch,plan.horizon_distance));
    // }

    else if (current_mode == io::Mode::idle) {
      // idle模式下降低发送频率（每10次循环发送一次，约500ms）
      if (++idle_counter >= 10) {
        io::Command command{false, false, 0, 0, 0};
        // 发送多次以提高数据到达率（中间可能有损耗）
        for (int i = 0; i < send_repeat_count; ++i) {
          cboard.send(command);
        }
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms); // 降低CPU占用
    }
    plotter.plot(data);
  }

  quit = true;
  if (detect_thread.joinable())
    detect_thread.join();
  if (plan_thread.joinable())
    plan_thread.join();
  io::Command command{false, false, 0, 0, 0};
  // 发送多次以提高数据到达率（中间可能有损耗）
  for (int i = 0; i < send_repeat_count; ++i) {
    cboard.send(command);
  }

  return 0;
}