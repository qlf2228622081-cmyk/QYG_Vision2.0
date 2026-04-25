#ifndef IO__CBOARD_HPP
#define IO__CBOARD_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "io/command.hpp"
#include "io/socketcan.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

// 四元数坐标系限域参数 (用于归一化数据还原)
#define q_min -1.0
#define q_max 1.0

/**
 * @brief 系统运行模式枚举
 */
enum Mode
{
  idle,       // 怠速/停止
  auto_aim,   // 自动对垒模式 (装甲板)
  small_buff, // 小符
  big_buff,   // 大符
  outpost     // 前哨站模式
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

/**
 * @brief 敌方颜色
 */
enum class EnemyColor
{
  red,
  blue
};

/**
 * @brief 哨兵专有开火模式
 */
enum ShootMode
{
  left_shoot,
  right_shoot,
  both_shoot
};
const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

/**
 * @brief 电控板通信接口类 (CBoard)
 * 逻辑：该类负责通过 SocketCAN 协议与电控底层进行双向通信。
 * 1. 接收底层发送的云台陀螺仪（IMU）四元数及其对应的时间戳。
 * 2. 提供时间戳插值功能 (imu_at)，确保视觉处理时的位姿与图像采集时刻精准对齐。
 * 3. 接收当前的比赛状态（模式、射速等）。
 * 4. 向电控下放视觉结算出的目标角度等指令。
 */
class CBoard
{
public:
  double bullet_speed; // 当前射速 (m/s)
  Mode mode;           // 当前视觉工作模式
  ShootMode shoot_mode;
  double ft_angle;     // 无人机/特殊兵种姿态角

  EnemyColor enemy_color() const;
  std::string enemy_color_string() const;

  /**
   * @brief 构造函数：初始化 CAN 接口并启动接收线程
   */
  CBoard(const std::string & config_path);

  /**
   * @brief 获取指定时间戳时刻的插值四元数
   * 逻辑：从 IMU 历史队列中搜索该时间前后的两帧数据，进行球面线性插值 (Slerp)。
   * 这是解决视觉处理链路延时、实现运动补偿的核心。
   */
  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  /**
   * @brief 发送视觉指令包
   */
  void send(Command command) const;

private:
  /**
   * @brief IMU 原始数据帧格式
   */
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  // 陀螺仪数据高频接收缓存队列 (Thread Safe)
  tools::ThreadSafeQueue<IMUData> queue_;

  // SocketCAN 驱动封装
  std::unique_ptr<SocketCAN> socketcan_;

  IMUData data_ahead_;  // 插值辅助点 A (较旧)
  IMUData data_behind_; // 插值辅助点 B (较新)

  int quaternion_canid_, bullet_speed_canid_, send_canid_; // 协议 ID 映射
  std::atomic<EnemyColor> enemy_color_;

  /**
   * @brief CAN 接收回调函数（由 SocketCAN 后台线程触发）
   */
  void callback(const can_frame & frame);
  
  /**
   * @brief 向 CAN 总线写入数据帧
   */
  void write_can_frame(const can_frame & frame) const;

  /**
   * @brief 检查 SocketCAN 接口是否正常可用
   */
  bool check_socketcan_available(const std::string & interface);
  
  /**
   * @brief 通信协议解码辅助：将接收到的整型还原为浮点数
   */
  float uint_to_float(int x_int, float x_min, float x_max, int bits);
};

}  // namespace io

#endif  // IO__CBOARD_HPP