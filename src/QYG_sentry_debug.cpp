#include <chrono>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
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
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_sentry.yaml | 位置参数,yaml配置文件路径 }";

using namespace std::chrono_literals;

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

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path,true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  // auto_buff::Buff_Detector buff_detector(config_path);
  // auto_buff::Solver buff_solver(config_path);
  // auto_buff::SmallTarget buff_small_target;
  // auto_buff::BigTarget buff_big_target;
  // auto_buff::Aimer buff_aimer(config_path);

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};
  int idle_counter = 0;  // idle模式下的计数器，用于降低发送频率

  // Debug模式相关变量
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  double current_fps = 0.0;
  cv::Mat debug_img;  // 用于显示的图像

  // 检测线程：异步进行图像采集和检测
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        camera.read(img, t);
        detector.push(img, t);  // 异步检测
      } else
        std::this_thread::sleep_for(10ms);
    }
  });

  // 规划线程：MPC规划和控制
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (mode.load() == io::GimbalMode::AUTO_AIM && !target_queue.empty()) {
        auto target = target_queue.pop();  // pop()会阻塞等待，但empty()检查可以避免在非自瞄模式下等待
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);

        // 直接使用Plan结构发送到Gimbal
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
          plan.pitch, plan.pitch_vel, plan.pitch_acc);
        tools::logger()->info("Sent Command - Control: {}, Fire: {}, Yaw: {:.2f}, Pitch: {:.2f}",
          plan.control, plan.fire, plan.yaw, plan.pitch);

        std::this_thread::sleep_for(10ms);
      } else
        std::this_thread::sleep_for(50ms);  // 减少等待时间，提高响应速度
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();
    auto current_mode = mode.load();  // 缓存模式值，避免重复调用load()

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", gimbal.str(current_mode));
      last_mode = current_mode;
    }

    /// 自瞄
    if (current_mode == io::GimbalMode::AUTO_AIM) {
      // 从检测队列获取结果（异步检测已完成）
      auto [img, armors, t] = detector.debug_pop();
      auto q = gimbal.q(t - 1ms);
      
      recorder.record(img, q, t);
      solver.set_R_gimbal2world(q);

      auto targets = tracker.track(armors, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      // Debug模式：绘制识别画面和信息
      debug_img = img;  // 这里不需要再次 clone，因为后续绘制操作不会影响 recorder 中的图像
      
      // 计算FPS
      auto now = std::chrono::steady_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
      fps_frame_count++;
      if (dt >= 1000) {  // 每秒更新一次FPS
        current_fps = fps_frame_count * 1000.0 / dt;
        fps_frame_count = 0;
        last_fps_time = now;
      }

      // 绘制检测到的装甲板
      for (const auto & armor : armors) {
        // 绘制装甲板四个角点
        if (!armor.points.empty()) {
          tools::draw_points(debug_img, armor.points, {0, 255, 0}, 2);
        }
        
        // 绘制装甲板信息
        std::string armor_info = fmt::format(
          "{:.2f} {} {} {}", armor.confidence, 
          auto_aim::COLORS[armor.color],
          auto_aim::ARMOR_NAMES[armor.name],
          auto_aim::ARMOR_TYPES[armor.type]);
        tools::draw_text(debug_img, armor_info, armor.center, {0, 255, 0}, 0.5, 1);
      }

      // 绘制目标跟踪信息
      if (!targets.empty()) {
        auto target = targets.front();
        
        // 绘制目标的所有装甲板位置（绿色）
        std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
        for (const Eigen::Vector4d & xyza : armor_xyza_list) {
          auto image_points =
            solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_img, image_points, {0, 255, 0}, 2);
        }

        // 绘制瞄准点（红色）
        Eigen::Vector4d aim_xyza = planner.debug_xyza;
        auto aim_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(debug_img, aim_points, {0, 0, 255}, 2);

        // 显示目标信息
        // Eigen::VectorXd x = target.ekf_x();
        // double distance = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);
        // std::string target_info = fmt::format(
        //   "Target: {} | Dist: {:.2f}m | Yaw: {:.1f}deg | W: {:.2f}rad/s",
        //   auto_aim::ARMOR_NAMES[target.name], distance, x[6] * 57.3, x[7]);
        // tools::draw_text(debug_img, target_info, {10, 150}, {255, 255, 0}, 0.6, 2);
      } else {
        // 没有目标时显示提示
        tools::draw_text(debug_img, "No Target", {10, 150}, {128, 128, 128}, 0.6, 2);
      }

      // 显示模式、FPS和装甲板数量
      int y_offset = 30;
      // std::string mode_text = fmt::format("Mode: {}", gimbal.str(current_mode));
      // tools::draw_text(debug_img, mode_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);
      
      // y_offset += 30;
      // std::string tracker_state_text = fmt::format("Tracker: {}", tracker.state());
      // tools::draw_text(debug_img, tracker_state_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);
      
      y_offset += 25;
      std::string fps_text = fmt::format("FPS: {:.1f}", current_fps);
      tools::draw_text(debug_img, fps_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);
      
      // y_offset += 30;
      // std::string armor_count_text = fmt::format("Armors: {}", armors.size());
      // tools::draw_text(debug_img, armor_count_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);

      // 显示云台和命令信息
      y_offset += 30;
      Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      std::string gimbal_text = fmt::format(
        "Gimbal: Yaw {:.1f}deg Pitch {:.1f}deg", 
        gimbal_pos[0] * 57.3, -gimbal_pos[1] * 57.3);
      tools::draw_text(debug_img, gimbal_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);

      // y_offset += 25;
      // auto gs = gimbal.state();
      // std::string bullet_text = fmt::format("Bullet Speed: {:.1f} m/s", gs.bullet_speed);
      // tools::draw_text(debug_img, bullet_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);


      // 显示图像（缩小尺寸以提高性能）
      cv::Mat display_img;
      cv::resize(debug_img, display_img, {}, 0.5, 0.5);
      cv::imshow("QYG Sentry Debug", display_img);
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) {  // 'q' 或 ESC 退出
        break;
      }
    }

    // /// 打符
    // else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
    //   buff_solver.set_R_gimbal2world(q);

    //   auto power_runes = buff_detector.detect(img);

    //   buff_solver.solve(power_runes);

    //   auto_aim::Plan buff_plan;
    //   if (mode.load() == io::GimbalMode::SMALL_BUFF) {
    //     buff_small_target.get_target(power_runes, t);
    //     auto target_copy = buff_small_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
    //     buff_big_target.get_target(power_runes, t);
    //     auto target_copy = buff_big_target;
    //     buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
    //   }
    //   gimbal.send(
    //     buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
    //     buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);
    // }
    
    else if (current_mode == io::GimbalMode::IDLE) {
      // idle模式下降低发送频率（每10次循环发送一次，约500ms）
      if (++idle_counter >= 10) {
        gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);  // 降低CPU占用
    }
  }

  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}


