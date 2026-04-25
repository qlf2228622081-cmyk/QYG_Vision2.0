#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{

/**
 * @brief 电控->视觉 串口数据包格式
 * 逻辑：底层电控板发送的反馈信息。
 */
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {'S', 'P'}; // 帧头
  uint8_t mode;                 // 视觉工作模式透传 (用于同步)
  float q[4];                   // 云台当前位姿四元数 (wxyz)
  float yaw;                    // 陀螺仪原始 Yaw (用于速度补偿计算)
  float pitch;                  // 陀螺仪原始 Pitch
  float bullet_speed;           // 射速反馈 (m/s)
  uint16_t crc16;               // 校验位
};

/**
 * @brief 视觉->电控 串口数据包格式
 * 逻辑：视觉系统结算出的控制指令。
 */
struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode;  // 0: 停止控制, 1: 跟随但不点放, 2: 跟随且开火
  float yaw;     // 目标绝对角度/相对偏移 (视协议而定)
  float pitch;
  uint16_t crc16;
};

enum class GimbalMode
{
  IDLE,        
  AUTO_AIM,    
  SMALL_BUFF,  
  BIG_BUFF     
};

struct GimbalState
{
  float yaw;
  float pitch;
  float bullet_speed;
};

/**
 * @brief 串口云台通信类 (Gimbal)
 * 逻辑：由于部分兵种（如英雄/步兵）采用串口直连而非 CAN 转换，
 * 该类封装了基于 C++ Serial 库的高速串口收发逻辑。
 * 1. 独立 read_thread 循环监听串口，并进行帧头比对与 CRC 校验。
 * 2. 具备四元数历史队列，支持视觉算法所需的位姿时间对齐。
 * 3. 具备自动断线重连逻辑。
 */
class Gimbal
{
public:
  /**
   * @brief 初始化串口，配置波特率 (921600)
   */
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;

  /**
   * @brief 根据时间戳插值获取云台位姿
   */
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  /**
   * @brief 发送控制量 (多参数版)
   */
  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  /**
   * @brief 结构体直发接口
   */
  void send(io::VisionToGimbal VisionToGimbal);

private:
  serial::Serial serial_; // 串口底层句柄

  std::thread thread_;    // 接收子线程
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;

  GimbalToVision rx_data_; // 接收缓冲区
  VisionToGimbal tx_data_; // 发送缓冲区

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  
  // 插值辅助用的时间戳序列
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000}; 

  bool read(uint8_t * buffer, size_t size);
  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP