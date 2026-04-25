#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/multithread/commandgener.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

/**
 * @brief 命令行参数解析规则
 * - help/h/usage/?: 输出帮助信息
 * - @config-path: YAML配置文件路径，默认指向英雄机器人配置
 */
const std::string keys =
  "{help h usage ? |      | 输出命令行参数说明}"
  "{@config-path   | configs/QYG_hero.yaml | 位置参数，yaml配置文件路径 }";

/**
 * @brief 视觉系统标准入口程序 (Standard Entry)
 * 
 * 详细逻辑流 (The standard vision closed-loop pipeline):
 * 1. 初始化 (Initialization):
 *    - 硬件层: 初始化工业相机驱动 (Camera) 和 电控串口/CAN通信 (CBoard)。
 *    - 算法层: 加载 YOLO 模型 (Detector)、建立位姿解算器 (Solver)、初始化多目标跟踪器 (Tracker)、
 *             弹道预测器 (Aimer) 以及 开火决策器 (Shooter)。
 * 2. 闭环主循环 (Execution Loop):
 *    a. 图像采集 (Image Acquisition): 从相机缓冲区读取最新一帧 Mat 图像，并获取其精确的硬件触发时间戳 t。
 *    b. 姿态对齐 (Pose Alignment): 利用时间戳 t 向电控请求该瞬间的 IMU 四元数姿态 q。
 *       注意: 此处减去 1ms 是为了补偿传感器链路的固定传输延迟，确保图像中的目标与当时的云台位置完美对应。
 *    c. 设定坐标参考 (Coordinate System): 将当前云台的 IMU 姿态注入 Solver，建立“图像-云台-世界”三者间的变换矩阵。
 *    d. 目标检测 (Object Detection): YOLO 推理识别图像中的所有潜在装甲板候选。
 *    e. 状态估计 (State Estimation): 
 *       - Solver 为每个候选装甲板计算 3D 空间坐标。
 *       - Tracker (内部含 EKF 卡尔曼滤波) 对观测结果进行时序平滑、ID 匹配和运动预测。
 *    f. 预测规划 (Ballistic & Prediction): Aimer 根据目标当前速度和距离，结合物理模型补偿弹道下坠和提前量，输出目标云台角度。
 *    g. 指令下放 (Execution): 将最终的控制指令通过 CAN 总线发送至底层电控板。
 */
int main(int argc, char * argv[])
{
  // 1. 解析命令行输入的配置文件路径
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 2. 初始化核心调试组件
  tools::Exiter exiter;     // 用于捕获 Ctrl+C 信号并优雅退出
  tools::Plotter plotter;   // 通用曲线绘制工具，辅助调参
  tools::Recorder recorder; // 高性能视频与元数据记录，用于复盘调试

  // 3. 初始化硬件通信层
  io::CBoard cboard(config_path); // 初始化电控板通信 (JSON/CAN/Serial)
  io::Camera camera(config_path); // 初始化工业相机 (海康/大恒/USB)

  // 4. 初始化算法模块
  auto_aim::YOLO detector(config_path, false);      // 装甲板检测神经网络
  auto_aim::Solver solver(config_path);             // 视觉位姿/重投影解算器
  auto_aim::Tracker tracker(config_path, solver);   // 多目标 EKF 跟踪器
  auto_aim::Aimer aimer(config_path);               // 弹道解算与瞄准预测
  auto_aim::Shooter shooter(config_path);           // 自动射击策略判断

  cv::Mat img;                                      // 图像容器
  Eigen::Quaterniond q;                             // 云台四元数姿态
  std::chrono::steady_clock::time_point t;          // 采集时间戳

  auto mode = io::Mode::idle;                       // 当前运行模式
  auto last_mode = io::Mode::idle;                  // 上一次模式，用于切换日志提示

  // 5. 进入视觉处理闭环
  while (!exiter.exit()) {
    // a. 采集环节: 获取图像及其时间戳
    if (!camera.read(img, t)) {
      continue;
    }
    
    // b. 姿态同步: 从姿态队列中获取与图像时间戳对齐的 IMU 姿态
    // 减去 1ms 是常用的经验值，用于抵消电控上传姿态与视觉接收图像间的时间轴微差
    q = cboard.imu_at(t - 1ms);
    mode = cboard.mode; // 获取电控发送的当前任务模式 (自瞄/打符/巡逻)

    // c. 模式切换显示
    if (last_mode != mode) {
      tools::logger()->info("视觉模式切换为: {}", io::MODES[mode]);
      last_mode = mode;
    }

    // d. 坐标映射: 设置云台当前位姿作为基准，用于后续的世界坐标转换
    solver.set_R_gimbal2world(q);

    // e. 识别环节: YOLO 模型推理
    auto armors = detector.detect(img);

    // f. 跟踪环节: EKF 状态滤波与多目标管理
    auto targets = tracker.track(armors, t);

    // g. 计算环节: Aimer 进行弹道下坠补偿、空气阻力修正及飞行时间提前量计算
    auto command = aimer.aim(targets, t, cboard.bullet_speed);

    // h. 指令发送: 下发最终的角度(yaw, pitch)和开火状态
    cboard.send(command);
  }

  return 0;
}