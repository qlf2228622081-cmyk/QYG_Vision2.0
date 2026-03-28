#include <chrono>
#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/cboard.hpp"
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
#include "tools/plotter.hpp"
// 在文件开头，定义共享变量（用于线程间传递命令角度）
std::atomic<double> last_cmd_yaw{0.0};
std::atomic<double> last_cmd_pitch{0.0};

const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数,yaml配置文件路径 }";

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
  tools::Plotter plotter;

  io::CBoard cboard(config_path);
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

  std::atomic<io::Mode> mode{io::Mode::idle};
  auto last_mode{io::Mode::idle};
  int idle_counter = 0;  // idle模式下的计数器，用于降低发送频率

  // Debug模式相关变量
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  double current_fps = 0.0;
  cv::Mat debug_img;  // 用于显示的图像
  
  // 性能监控变量
  struct PerformanceStats {
    double debug_pop_ms = 0.0;
    double imu_ms = 0.0;
    double recorder_ms = 0.0;
    double tracker_ms = 0.0;
    double draw_ms = 0.0;
    double imshow_ms = 0.0;
    double total_ms = 0.0;
  } perf_stats;
  
  auto last_perf_log = std::chrono::steady_clock::now();
  int perf_log_counter = 0;
  
  nlohmann::json data;

  // 检测线程：异步进行图像采集和检测
  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!quit && !exiter.exit()) {
      if (mode.load() == io::Mode::auto_aim) {
        camera.read(img, t);
        detector.push(img, t);  // 异步检测
      } else
        //std::this_thread::sleep_for(10ms);
        continue;
    }
  });

  // 规划线程：MPC规划和控制
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (mode.load() == io::Mode::auto_aim && !target_queue.empty()) {
        auto target = target_queue.pop();  // pop()会阻塞等待，但empty()检查可以避免在非自瞄模式下等待
        auto plan = planner.plan(target, cboard.bullet_speed);

        // 使用初始化列表构造Command
        io::Command command{
          plan.control, plan.fire, plan.yaw, plan.pitch, 0  // horizon_distance设为0
        };
        cboard.send(command);
        last_cmd_yaw.store(plan.yaw);
        last_cmd_pitch.store(plan.pitch);
        tools::logger()->info("Sent Command - Control: {}, Fire: {}, Yaw: {:.2f}, Pitch: {:.2f}",
          command.control, command.shoot, command.yaw, command.pitch);

        std::this_thread::sleep_for(10ms);
      } else
        std::this_thread::sleep_for(50ms);  // 减少等待时间，提高响应速度
    }
  });

  while (!exiter.exit()) {
    mode = cboard.mode;
    auto current_mode = mode.load();

    if (last_mode != current_mode) {
      tools::logger()->info("Switch to {}", io::MODES[current_mode]);
      last_mode = current_mode;
    }

    /// 自瞄
    if (current_mode == io::Mode::auto_aim) {
      auto frame_start = std::chrono::steady_clock::now();
      
      // 从检测队列获取结果（异步检测已完成）
      auto t0 = std::chrono::steady_clock::now();
      auto [img, armors, t] = detector.debug_pop();
      auto t1 = std::chrono::steady_clock::now();
      perf_stats.debug_pop_ms = tools::delta_time(t1, t0) * 1000.0;
      
      auto t2 = std::chrono::steady_clock::now();
      auto q = cboard.imu_at(t);
      auto t3 = std::chrono::steady_clock::now();
      perf_stats.imu_ms = tools::delta_time(t3, t2) * 1000.0;
      
      auto t4 = std::chrono::steady_clock::now();
      recorder.record(img, q, t);
      auto t5 = std::chrono::steady_clock::now();
      perf_stats.recorder_ms = tools::delta_time(t5, t4) * 1000.0;
      
      solver.set_R_gimbal2world(q);

      auto t6 = std::chrono::steady_clock::now();
      auto targets = tracker.track(armors, t);
      auto t7 = std::chrono::steady_clock::now();
      perf_stats.tracker_ms = tools::delta_time(t7, t6) * 1000.0;
      
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      // Debug模式：绘制识别画面和信息
      debug_img = img;  // 这里不需要再次 clone，因为后续绘制操作不会影响 recorder 中的图像
      
      auto t8 = std::chrono::steady_clock::now();
      
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

      auto t9 = std::chrono::steady_clock::now();
      perf_stats.draw_ms = tools::delta_time(t9, t8) * 1000.0;

      // 显示模式、FPS和性能信息
      int y_offset = 30;
      // std::string mode_text = fmt::format("Mode: {}", io::MODES[current_mode]);
      // tools::draw_text(debug_img, mode_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);
      
      // y_offset += 30;
      // std::string tracker_state_text = fmt::format("Tracker: {}", tracker.state());
      // tools::draw_text(debug_img, tracker_state_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);
      
      y_offset += 25;
      std::string fps_text = fmt::format("FPS: {:.1f}", current_fps);
      tools::draw_text(debug_img, fps_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);
      
      // 显示性能信息
      y_offset += 25;
      std::string perf_text = fmt::format("Pop:{:.1f}ms IMU:{:.1f}ms Rec:{:.1f}ms Trk:{:.1f}ms",
        perf_stats.debug_pop_ms, perf_stats.imu_ms, perf_stats.recorder_ms, perf_stats.tracker_ms);
      tools::draw_text(debug_img, perf_text, {10, y_offset}, {255, 255, 0}, 0.5, 1);
      
      y_offset += 20;
      std::string perf_text2 = fmt::format("Draw:{:.1f}ms ImShow:{:.1f}ms Total:{:.1f}ms",
        perf_stats.draw_ms, perf_stats.imshow_ms, perf_stats.total_ms);
      tools::draw_text(debug_img, perf_text2, {10, y_offset}, {255, 255, 0}, 0.5, 1);
      
      // y_offset += 30;
      // std::string armor_count_text = fmt::format("Armors: {}", armors.size());
      // tools::draw_text(debug_img, armor_count_text, {10, y_offset}, {255, 255, 255}, 0.7, 2);

      // 显示云台和命令信息
      y_offset += 25;
      Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);
      std::string gimbal_text = fmt::format(
        "Gimbal: Yaw {:.1f}deg Pitch {:.1f}deg", 
        gimbal_pos[0] * 57.3, -gimbal_pos[1] * 57.3);
      tools::draw_text(debug_img, gimbal_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);
      
      // ========== 添加PlotJuggler数据 ==========
      // 准备数据
      nlohmann::json plot_data;

      // 命令角度（从规划线程获取）
      double cmd_yaw = last_cmd_yaw.load();
      double cmd_pitch = last_cmd_pitch.load();

      // 实际角度（从IMU计算）
      double actual_yaw = gimbal_pos[0];
      double actual_pitch = -gimbal_pos[1];  // 注意pitch的符号

      // 计算误差（转换为度）
      double yaw_error = (cmd_yaw - actual_yaw) * 57.3;      // 度
      double pitch_error = (cmd_pitch - actual_pitch) * 57.3; // 度

      // 添加到plot数据（PlotJuggler会接收这些字段）
      plot_data["cmd_yaw"] = cmd_yaw * 57.3;           // 命令yaw（度）
      plot_data["cmd_pitch"] = cmd_pitch * 57.3;       // 命令pitch（度）
      plot_data["actual_yaw"] = actual_yaw * 57.3;     // 实际yaw（度）
      plot_data["actual_pitch"] = actual_pitch * 57.3; // 实际pitch（度）
      plot_data["yaw_error"] = yaw_error;              // yaw误差（度）
      plot_data["pitch_error"] = pitch_error;          // pitch误差（度）

      // 可选：添加时间戳（PlotJuggler自动处理，但可以添加相对时间）
      static auto plot_start_time = std::chrono::steady_clock::now();
      auto plot_time = std::chrono::steady_clock::now();
      plot_data["t"] = tools::delta_time(plot_time, plot_start_time);

      // 发送到PlotJuggler
      plotter.plot(plot_data);
      // ==========================================
      

      // y_offset += 25;
      // std::string bullet_text = fmt::format("Bullet Speed: {:.1f} m/s", cboard.bullet_speed);
      // tools::draw_text(debug_img, bullet_text, {10, y_offset}, {255, 255, 255}, 0.6, 1);
      
      // std::vector<cv::Point3f> worldPoints;
      // float HEIGHT   = -0.5;
      // float GRID_NUM = 50;
      // float GRID_SIZE = 0.1;
    
      // for (int x = 0; x < GRID_NUM; ++x) {          // X方向：0 ~ GRID_NUM-1
      //   for (int y = 0; y < GRID_NUM; ++y) {        // Y方向：0 ~ GRID_NUM-1
      //     // 计算X坐标：列索引 × 网格边长（从左到右递增）
      //     double x_coord = x * GRID_SIZE;
    
      //     // 计算Y坐标：原始坐标 - 偏移量（实现居中）
      //     double y_coord = y * GRID_SIZE - (GRID_NUM * GRID_SIZE) / 2.0;
    
      //     // 计算Z坐标：固定负值（与Python保持一致，可根据需求修改符号）
      //     double z_coord = HEIGHT;
    
      //     // 将当前点添加到点云容器中
      //     worldPoints.emplace_back(x_coord, y_coord, z_coord);
      //   }
      // }
    
      // for (int x1 = 0; x1 < GRID_NUM + 20; ++x1) {
      //   double x_coord = x1 * GRID_SIZE;
      //   double y_coord = 0;
      //   double z_coord = HEIGHT;
      //   worldPoints.emplace_back(x_coord, y_coord, z_coord);
      // }
    
      // std::vector<cv::Point2f> pixel_point = solver.world2pixel(worldPoints);
    
      // int H = debug_img.rows;  // 高度（行数）
      // int W = debug_img.cols;  // 宽度（列数）
    
      // for (const auto &projected_point : pixel_point) {
      //   int x = static_cast<int>(projected_point.x);
      //   int y = static_cast<int>(projected_point.y);
      //   if (x < 0 || x > W || y < 0 || y > H) continue;
      //   cv::Point point(x, y);
      //   tools::draw_point(debug_img, point, {255, 255, 255}, 1);
      // }

      // 显示图像（缩小尺寸以提高性能）
      auto t10 = std::chrono::steady_clock::now();
      cv::Mat display_img;
      cv::resize(debug_img, display_img, {}, 0.5, 0.5);
      
      cv::imshow("QYG Hero Debug", display_img);
      auto t11 = std::chrono::steady_clock::now();
      perf_stats.imshow_ms = tools::delta_time(t11, t10) * 1000.0;
      
      auto frame_end = std::chrono::steady_clock::now();
      perf_stats.total_ms = tools::delta_time(frame_end, frame_start) * 1000.0;
      
      // 定期输出性能日志（每30帧一次）
      perf_log_counter++;
      if (perf_log_counter >= 30) {
        tools::logger()->info(
          "[性能] FPS:{:.1f} | Pop:{:.1f}ms | IMU:{:.1f}ms | Rec:{:.1f}ms | Trk:{:.1f}ms | Draw:{:.1f}ms | ImShow:{:.1f}ms | Total:{:.1f}ms",
          current_fps, perf_stats.debug_pop_ms, perf_stats.imu_ms, perf_stats.recorder_ms,
          perf_stats.tracker_ms, perf_stats.draw_ms, perf_stats.imshow_ms, perf_stats.total_ms);
        perf_log_counter = 0;
      }
      
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) {  // 'q' 或 ESC 退出
        break;
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
      // idle模式下降低发送频率（每10次循环发送一次，约500ms）
      if (++idle_counter >= 10) {
        io::Command command{false, false, 0, 0, 0};
        cboard.send(command);
        idle_counter = 0;
      }
      std::this_thread::sleep_for(50ms);  // 降低CPU占用
    }
  }

  quit = true;
  if (detect_thread.joinable()) detect_thread.join();
  if (plan_thread.joinable()) plan_thread.join();
  io::Command command{false, false, 0, 0, 0};
  cboard.send(command);



  return 0;
}