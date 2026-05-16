#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "io/camera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml   | yaml配置文件路径}"
  "{tradition t    | false                   | 是否使用传统视觉识别}"
  "{bullet-speed b | 23.0                    | 仅用于预计打击点计算的弹速}"
  "{fps            | 30.0                    | 保存视频帧率}"
  "{output o       | records/only_detect.avi | 绘制后视频保存路径}"
  "{display d      |                         | 是否显示实时窗口}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("@config-path");
  auto use_tradition = cli.get<bool>("tradition");
  auto bullet_speed = cli.get<double>("bullet-speed");
  auto fps = cli.get<double>("fps");
  auto output_path = cli.get<std::string>("output");
  auto display = cli.has("display");

  tools::Exiter exiter;
  tools::Recorder recorder(fps, "only_detect_raw");

  tools::logger()->info("only_detect_and_records 启动");
  tools::logger()->info("配置文件: {}", config_path);
  tools::logger()->info("绘制后视频: {}", output_path);
  tools::logger()->info("Recorder 原始回放: records/*_only_detect_raw.avi 和 records/*_only_detect_raw.txt");
  tools::logger()->info("识别模式: {}", use_tradition ? "traditional" : "yolo");
  tools::logger()->info("预计打击弹速: {:.1f} m/s", bullet_speed);
  tools::logger()->info("显示窗口: {}", display ? "on" : "off");
  tools::logger()->info("停止方式: 终端 Ctrl+C；如果开启 -d，也可以在窗口中按 q 或 Esc");

  auto yaml = tools::load(config_path);
  auto enemy_color = tools::read<std::string>(yaml, "enemy_color");

  io::Camera camera(config_path);
  tools::logger()->info("相机对象初始化完成，正在等待首帧...");
  auto_aim::Detector detector(config_path, true);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  tracker.set_enemy_color(enemy_color);
  auto_aim::Aimer aimer(config_path);
  tools::logger()->info("敌方颜色过滤: {}", enemy_color);

  const Eigen::Quaterniond fixed_imu_q = Eigen::Quaterniond::Identity();

  cv::VideoWriter debug_writer;
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  int total_frame_count = 0;
  double current_fps = 0.0;
  std::size_t last_armor_count = 0;
  double last_detect_ms = 0.0;
  int empty_frame_counter = 0;

  while (!exiter.exit()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    camera.read(img, timestamp);
    if (img.empty()) {
      empty_frame_counter++;
      if (empty_frame_counter % 20 == 0) {
        tools::logger()->warn("仍未拿到有效图像（{}次），请检查相机连接/曝光/网口。", empty_frame_counter);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    empty_frame_counter = 0;

    recorder.record(img, fixed_imu_q, timestamp);

    auto detect_start = std::chrono::steady_clock::now();
    std::list<auto_aim::Armor> armors;
    if (use_tradition)
      armors = detector.detect(img, total_frame_count);
    else
      armors = yolo.detect(img, total_frame_count);
    auto detect_end = std::chrono::steady_clock::now();
    last_detect_ms = tools::delta_time(detect_end, detect_start) * 1000.0;

    solver.set_R_gimbal2world(fixed_imu_q);
    auto targets = tracker.track(armors, timestamp);
    last_armor_count = armors.size();
    if (!targets.empty()) {
      aimer.aim(targets, timestamp, bullet_speed, true);
    }

    cv::Mat debug_img = img.clone();

    // 1) 蓝色：YOLO（或传统）当前选中的1个检测框
    if (!armors.empty()) {
      const auto & armor = armors.front();
      if (!armor.points.empty()) {
        tools::draw_points(debug_img, armor.points, {255, 0, 0}, 2);
      }
      auto name_id = static_cast<std::size_t>(armor.name);
      auto color_id = static_cast<std::size_t>(armor.color);
      auto type_id = static_cast<std::size_t>(armor.type);
      std::string armor_info = fmt::format(
        "{:.2f} {} {} {}", armor.confidence,
        color_id < auto_aim::COLORS.size() ? auto_aim::COLORS[color_id] : "unknown",
        name_id < auto_aim::ARMOR_NAMES.size() ? auto_aim::ARMOR_NAMES[name_id] : "unknown",
        type_id < auto_aim::ARMOR_TYPES.size() ? auto_aim::ARMOR_TYPES[type_id] : "unknown");
      tools::draw_text(debug_img, armor_info, armor.center, {255, 0, 0}, 0.5, 1);
    }

    // 2) 绿色：EKF 预测装甲板框（常见为4个）
    if (!targets.empty()) {
      const auto target = targets.front();
      int ekf_box_count = 0;
      for (const auto & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(debug_img, image_points, {0, 255, 0}, 2);
        ekf_box_count++;
        if (ekf_box_count >= 4) break;
      }

      // 3) 红色：预计打击框（1个）
      auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        auto aim_points =
          solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
        tools::draw_points(debug_img, aim_points, {0, 0, 255}, 2);
      }
    }

    fps_frame_count++;
    total_frame_count++;
    auto now = std::chrono::steady_clock::now();
    auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
    if (dt_ms >= 1000) {
      current_fps = fps_frame_count * 1000.0 / dt_ms;
      tools::logger()->info(
        "运行中 | frame:{} | fps:{:.1f} | armors:{} | detect:{:.1f}ms | output:{}",
        total_frame_count, current_fps, last_armor_count, last_detect_ms, output_path);
      fps_frame_count = 0;
      last_fps_time = now;
    }

    tools::draw_text(
      debug_img,
      fmt::format(
        "FPS:{:.1f} Detect:{:.1f}ms Armors:{} Targets:{} Mode:{}",
        current_fps, last_detect_ms, armors.size(), targets.size(), use_tradition ? "traditional" : "yolo"),
      {10, 30}, {255, 255, 255}, 0.7, 2);

    if (!debug_writer.isOpened()) {
      auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      debug_writer.open(output_path, fourcc, fps, debug_img.size());
      if (!debug_writer.isOpened()) {
        tools::logger()->error("无法打开视频保存路径: {}", output_path);
        return 1;
      }
      tools::logger()->info("绘制后视频保存到: {}", output_path);
    }
    debug_writer.write(debug_img);

    if (display) {
      cv::Mat display_img;
      cv::resize(debug_img, display_img, {}, 0.5, 0.5);
      cv::imshow("Only Detect And Records", display_img);
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) break;
    } else {
      auto key = cv::waitKey(1);
      if (key == 'q' || key == 27) break;
    }
  }

  debug_writer.release();
  cv::destroyAllWindows();
  tools::logger()->info("only_detect_and_records 已停止，共处理 {} 帧。", total_frame_count);
  return 0;
}
