#ifndef IO__MINDVISION_HPP
#define IO__MINDVISION_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "CameraApi.h"
#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

/**
 * @brief 迈德威视工业相机驱动类 (MindVision)
 * 逻辑：基于迈德威视官方 CameraSDK 封装。采用了与海康类相似的“异步取图 + 守护进程”架构，
 * 专为 Robomaster 比赛现场的高频掉线环境设计了 USB 硬件层重置逻辑。
 */
class MindVision : public CameraBase
{
public:
  /**
   * @brief 构造函数
   * @param exposure_ms 曝光时间 (ms)
   * @param gamma 伽马校正系数
   * @param vid_pid 设备的 VID:PID
   */
  MindVision(double exposure_ms, double gamma, const std::string & vid_pid);

  /**
   * @brief 析构函数
   */
  ~MindVision() override;

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

  double exposure_ms_, gamma_;
  CameraHandle handle_;              // SDK 设备句柄
  int height_, width_;               // 相机原始分辨率
  bool quit_, ok_;                   // 状态与退出标志

  std::thread capture_thread_;       // 取图线程
  std::thread daemon_thread_;        // 守护重连线程
  tools::ThreadSafeQueue<CameraData> queue_; // 缓存队列

  int vid_, pid_; // 硬件特征值，供 libusb 使用

  /**
   * @brief 启动 SDK 采集流程
   */
  void open();

  /**
   * @brief 异常处理封装好的开启函数
   */
  void try_open();

  /**
   * @brief 停止采集并释放 SDK 资源
   */
  void close();

  void set_vid_pid(const std::string & vid_pid);

  /**
   * @brief 触发物理级 USB 重置 (针对软连接锁死)
   */
  void reset_usb() const;
};

}  // namespace io

#endif  // IO__MINDVISION_HPP