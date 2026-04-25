#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "MvCameraControl.h"
#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

/**
 * @brief 海康机器人工业相机驱动类 (HikRobot)
 * 逻辑：基于海康官方 MVS SDK 封装，支持 USB 和 GigE (网口) 两种连接方式。
 * 核心特性：
 * 1. 采用单例/守护线程架构，支持硬件故障后的 USB 自动重置与驱动重连。
 * 2. 异步取图逻辑，通过独立线程进行 Bayer 到 RGB 的色彩空间转换。
 * 3. 硬件级的定时触发或连续采集。
 */
class HikRobot : public CameraBase
{
public:
  /**
   * @brief USB 相机构造函数
   * @param vid_pid 设备的 VendorID:ProductID (如 "1234:5678")
   */
  HikRobot(double exposure_ms, double gain, const std::string & vid_pid);
  
  /**
   * @brief GigE (网口) 相机构造函数
   */
  HikRobot(double exposure_ms, double gain);

  /**
   * @brief 析构函数：关闭采集并清理 SDK 句柄
   */
  ~HikRobot() override;

  /**
   * @brief 读取数据接口 (实现基类)
   */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_us_; // 内部曝光时间 (us)
  double gain_;        // 硬件增益

  std::thread daemon_thread_;         // 守护线程，判断心跳与重连
  std::atomic<bool> daemon_quit_{false};

  void * handle_;                     // 海康 SDK 设备句柄
  std::thread capture_thread_;        // 图像采集子线程
  std::atomic<bool> capturing_{false}; // 状态位：正在抓图
  std::atomic<bool> capture_quit_{false};
  tools::ThreadSafeQueue<CameraData> queue_; // 异步图像缓存

  int vid_, pid_; // 硬件标识，用于故障重置 USB 端口

  /**
   * @brief 启动 USB 抓图流程 (Open -> Config -> Start)
   */
  void capture_start();

  /**
   * @brief 启动 GigE (网口) 抓图流程
   */
  void capture_start_GigE();

  /**
   * @brief 停止采集并释放句柄
   */
  void capture_stop();

  // SDK 参数设置辅助函数
  void set_float_value(const std::string & name, double value);
  void set_enum_value(const std::string & name, unsigned int value);

  void set_vid_pid(const std::string & vid_pid);
  /**
   * @brief 调用 libusb 对特定端口进行硬件级重启 (针对 USB 掉线)
   */
  void reset_usb() const;
};

}  // namespace io

#endif  // IO__HIKROBOT_HPP