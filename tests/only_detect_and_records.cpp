#include <fmt/core.h>

#include <Eigen/Geometry>
#include <chrono>
#include <memory>
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

// 命令行参数定义字符串（OpenCV 的 CommandLineParser 使用这种格式描述参数）。
// 每一项格式为：{参数名 | 默认值 | 参数说明}
// 其中带 @ 前缀的是“位置参数”（必填或按位置传入），其余是可选参数。
const std::string keys =
  "{help h usage ? |                         | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml   | yaml配置文件路径}"
  "{tradition t    | false                   | 是否使用传统视觉识别}"
  "{bullet-speed b | 23.0                    | 仅用于预计打击点计算的弹速}"
  "{video v        |                         | 本地视频路径；传入后将使用视频而不是相机}"
  "{fps            | 30.0                    | 保存视频帧率}"
  "{output o       | records/only_detect.avi | 绘制后视频保存路径}"
  "{display d      |                         | 是否显示实时窗口}";

int main(int argc, char * argv[])
{
  // 1) 解析命令行参数。
  // argc/argv 来自 main 入口，keys 描述了支持的参数及默认值。
  cv::CommandLineParser cli(argc, argv, keys);
  // 如果用户请求帮助，或者没有提供必须的配置路径，则打印帮助并退出。
  // 注意：@config-path 设置了默认值，这里保留原逻辑用于防御性检查。
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 2) 读取并缓存命令行参数，后续直接使用局部变量，避免多次查询 parser。
  auto config_path = cli.get<std::string>("@config-path");
  auto use_tradition = cli.get<bool>("tradition");
  auto bullet_speed = cli.get<double>("bullet-speed");
  auto video_path = cli.get<std::string>("video");
  auto fps = cli.get<double>("fps");
  auto output_path = cli.get<std::string>("output");
  // display 参数是“开关型”，只要出现即视为 true。
  auto display = cli.has("display");
  auto use_video_input = !video_path.empty();

  // 3) 创建退出控制器与原始录制器。
  // Exiter 通常用于接收 Ctrl+C 等退出信号。
  // Recorder 用于保存“原始图像 + 时间戳 + IMU”等回放数据，便于离线复盘。
  tools::Exiter exiter;
  tools::Recorder recorder(fps, "only_detect_raw");

  // 4) 输出启动日志，便于运行时确认配置与行为模式。
  tools::logger()->info("only_detect_and_records 启动");
  tools::logger()->info("配置文件: {}", config_path);
  tools::logger()->info("绘制后视频: {}", output_path);
  tools::logger()->info("Recorder 原始回放: records/*_only_detect_raw.avi 和 records/*_only_detect_raw.txt");
  tools::logger()->info("识别模式: {}", use_tradition ? "traditional" : "yolo");
  tools::logger()->info("输入源: {}", use_video_input ? fmt::format("video({})", video_path) : "camera");
  tools::logger()->info("预计打击弹速: {:.1f} m/s", bullet_speed);
  tools::logger()->info("显示窗口: {}", display ? "on" : "off");
  tools::logger()->info("停止方式: 终端 Ctrl+C；如果开启 -d，也可以在窗口中按 q 或 Esc");

  // 5) 读取配置，获取敌方颜色（用于过滤目标阵营）。
  auto yaml = tools::load(config_path);
  auto enemy_color = tools::read<std::string>(yaml, "enemy_color");

  // 6) 初始化全链路模块：
  // - Camera：图像输入
  // - Detector / YOLO：两套识别器（按命令行选择）
  // - Solver：坐标解算与重投影
  // - Tracker：时序跟踪（内部常含滤波器）
  // - Aimer：根据目标状态计算预计击打点
  std::unique_ptr<io::Camera> camera;
  cv::VideoCapture video_cap;
  if (use_video_input) {
    video_cap.open(video_path);
    if (!video_cap.isOpened()) {
      tools::logger()->error("无法打开本地视频: {}", video_path);
      return 1;
    }
    tools::logger()->info("本地视频打开成功: {}", video_path);
  } else {
    camera = std::make_unique<io::Camera>(config_path);
    tools::logger()->info("相机对象初始化完成，正在等待首帧...");
  }
  auto_aim::Detector detector(config_path, true);
  auto_aim::YOLO yolo(config_path, true);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  tracker.set_enemy_color(enemy_color);
  auto_aim::Aimer aimer(config_path);
  tools::logger()->info("敌方颜色过滤: {}", enemy_color);

  // 当前测试固定使用单位四元数，表示“云台坐标系与世界坐标系不旋转”。
  // 作用：为 solver/tracker 提供一个稳定的姿态输入，便于只验证视觉链路。
  const Eigen::Quaterniond fixed_imu_q = Eigen::Quaterniond::Identity();

  // 调试视频写入器（写入的是“绘制后的可视化帧”，不是原始帧）。
  cv::VideoWriter debug_writer;
  // 以下变量用于实时统计性能与状态。
  auto last_fps_time = std::chrono::steady_clock::now();
  int fps_frame_count = 0;
  int total_frame_count = 0;
  double current_fps = 0.0;
  std::size_t last_armor_count = 0;
  double last_detect_ms = 0.0;
  int empty_frame_counter = 0;

  // 7) 主循环：持续采集 -> 识别 -> 跟踪 -> 解算/瞄准 -> 绘制 -> 保存/显示，直到收到退出信号。
  while (!exiter.exit()) {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    // 根据输入源读取一帧图像及对应时间戳。
    if (use_video_input) {
      if (!video_cap.read(img)) {
        tools::logger()->info("视频读取结束或失败，程序即将退出。");
        break;
      }
      timestamp = std::chrono::steady_clock::now();
    } else {
      camera->read(img, timestamp);
    }
    // 空帧防御：相机未就绪、链路异常、曝光异常等都可能导致空图。
    if (img.empty()) {
      empty_frame_counter++;
      // 每 20 次空帧打一条告警，避免日志刷屏同时保留可观测性。
      if (empty_frame_counter % 20 == 0) {
        tools::logger()->warn("仍未拿到有效图像（{}次），请检查相机连接/曝光/网口。", empty_frame_counter);
      }
      // 适当 sleep 防止空转占满 CPU。
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    // 一旦拿到有效帧，重置空帧计数。
    empty_frame_counter = 0;

    // 记录原始帧（用于离线回放，不受后续绘制影响）。
    recorder.record(img, fixed_imu_q, timestamp);

    // 8) 检测阶段计时：统计单帧识别耗时（毫秒）。
    auto detect_start = std::chrono::steady_clock::now();
    std::list<auto_aim::Armor> armors;
    // 根据命令行选择传统识别或 YOLO 识别。
    // total_frame_count 可用于某些模块内部做帧号相关逻辑。
    if (use_tradition)
      armors = detector.detect(img, total_frame_count);
    else
      armors = yolo.detect(img, total_frame_count);
    auto detect_end = std::chrono::steady_clock::now();
    last_detect_ms = tools::delta_time(detect_end, detect_start) * 1000.0;

    solver.set_R_gimbal2world(fixed_imu_q);
    auto targets = tracker.track(armors, timestamp);
    last_armor_count = armors.size();
    // 有可跟踪目标时，计算预计击打点（第三个参数是弹速）。
    if (!targets.empty()) {
      // 第四个参数 true 一般表示启用调试输出/调试路径（取决于 Aimer 实现）。
      aimer.aim(targets, timestamp, bullet_speed, true);
    }

    // 复制原图作为调试画布，所有可视化都画在 debug_img 上。
    cv::Mat debug_img = img.clone();

    // 1) 蓝色：YOLO（或传统）当前选中的1个检测框
    if (!armors.empty()) {
      // 这里仅展示 armors.front()，用于快速观察“当前主检测结果”。
      const auto & armor = armors.front();
      if (!armor.points.empty()) {
        // armor.points 通常是装甲板四角点（或用于外接框的关键点）。
        tools::draw_points(debug_img, armor.points, {255, 0, 0}, 2);
      }
      // 枚举值转索引前先转 size_t，后续配合越界判断安全取名。
      auto name_id = static_cast<std::size_t>(armor.name);
      auto color_id = static_cast<std::size_t>(armor.color);
      auto type_id = static_cast<std::size_t>(armor.type);
      // 组织识别标签文本：置信度 + 颜色 + 名称 + 类型。
      // 若索引越界则回退到 unknown，避免非法访问。
      std::string armor_info = fmt::format(
        "{:.2f} {} {} {}", armor.confidence,
        color_id < auto_aim::COLORS.size() ? auto_aim::COLORS[color_id] : "unknown",
        name_id < auto_aim::ARMOR_NAMES.size() ? auto_aim::ARMOR_NAMES[name_id] : "unknown",
        type_id < auto_aim::ARMOR_TYPES.size() ? auto_aim::ARMOR_TYPES[type_id] : "unknown");
      tools::draw_text(debug_img, armor_info, armor.center, {255, 0, 0}, 0.5, 1);
    }

    // 2) 绿色：EKF 预测装甲板框（常见为4个）
    if (!targets.empty()) {
      // 同样只展示首个主目标，降低画面复杂度。
      const auto target = targets.front();
      int ekf_box_count = 0;
      // 遍历跟踪器给出的候选装甲板状态（位置+朝向），重投影回图像坐标绘制。
      for (const auto & xyza : target.armor_xyza_list()) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(debug_img, image_points, {0, 255, 0}, 2);
        ekf_box_count++;
        // 最多画 4 个，避免过多候选影响可读性。
        if (ekf_box_count >= 4) break;
      }

      // 3) 红色：预计打击框（1个）
      auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        // 将预计击打姿态同样重投影到图像，红框用于观察“最终瞄准点”是否合理。
        auto aim_points =
          solver.reproject_armor(aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
        tools::draw_points(debug_img, aim_points, {0, 0, 255}, 2);
      }
    }

    // 9) FPS 统计（按 1 秒窗口更新一次）。
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

    // 将关键运行指标直接叠加到画面左上角，便于录屏回放时快速定位问题。
    tools::draw_text(
      debug_img,
      fmt::format(
        "FPS:{:.1f} Detect:{:.1f}ms Armors:{} Targets:{} Mode:{}",
        current_fps, last_detect_ms, armors.size(), targets.size(), use_tradition ? "traditional" : "yolo"),
      {10, 30}, {255, 255, 255}, 0.7, 2);

    // 10) 延迟初始化视频写入器：
    // 必须等拿到第一帧后，才能确定输出分辨率（debug_img.size()）。
    if (!debug_writer.isOpened()) {
      auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
      debug_writer.open(output_path, fourcc, fps, debug_img.size());
      if (!debug_writer.isOpened()) {
        tools::logger()->error("无法打开视频保存路径: {}", output_path);
        return 1;
      }
      tools::logger()->info("绘制后视频保存到: {}", output_path);
    }
    // 将本帧可视化结果写入输出视频。
    debug_writer.write(debug_img);

    // 11) 显示逻辑：
    // - 开启 display：显示缩小后的窗口，降低显示开销；
    // - 未开启 display：仍调用 waitKey(1) 以保持按键退出路径一致。
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

  // 12) 退出前释放资源，确保视频文件索引完整写回。
  debug_writer.release();
  cv::destroyAllWindows();
  tools::logger()->info("only_detect_and_records 已停止，共处理 {} 帧。", total_frame_count);
  return 0;
}
