#ifndef IO__USBCamera_HPP
#define IO__USBCamera_HPP

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace io
{

/**
 * @brief 通用 USB 相机驱动类 (USBCamera)
 * 逻辑：针对非 SDK 驱动的 UVC 协议相机。
 * 采用了“异步取图 + 自动重连”的健壮性架构：
 * 1. 内部 capture_thread_ 不间断从 OpenCV VideoCapture 读取图片并压入缓存队列。
 * 2. 内部 daemon_thread_ 负责监控硬件连接，若掉线则尝试自动重新开启相机。
 * 3. 利用 ThreadSafeQueue 确保多感知任务取图时不发生阻塞。
 */
class USBCamera
{
public:
  /**
   * @brief 构造函数
   * @param open_name 设备挂载名称 (如 "video0")
   * @param config_path 配置文件路径 (用于读取分辨率、曝光等)
   */
  USBCamera(const std::string & open_name, const std::string & config_path);
  
  /**
   * @brief 析构函数：优雅停止子线程并释放硬件资源
   */
  ~USBCamera();

  /**
   * @brief [同步接口] 直接读取当前帧图像 (会发生系统调用阻塞)
   */
  cv::Mat read();

  /**
   * @brief [异步接口] 从内部平滑队列中读取图像及对应的时间戳 (推荐)
   */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

  std::string device_name; // 相机角色名称 (如 "left", "right")

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::mutex cap_mutex_;
  cv::VideoCapture cap_;
  cv::Mat img_;
  std::string open_name_;
  
  // 相机控制参数
  int usb_exposure_, usb_frame_rate_, sharpness_;
  double image_width_, image_height_;
  int usb_gamma_, usb_gain_;
  
  int open_count_;     // 重连尝试计数
  bool quit_, ok_;     // 线程生命周期与状态标志

  std::thread capture_thread_; // 取图子线程
  std::thread daemon_thread_;  // 守护 (自动重连) 子线程
  tools::ThreadSafeQueue<CameraData> queue_; // 帧数据缓存队列

  void try_open();
  void open();
  void close();
};

}  // namespace io

#endif