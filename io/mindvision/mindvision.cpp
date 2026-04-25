#include "mindvision.cpp"

#include <libusb-1.0/libusb.h>
#include <stdexcept>
#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{

/**
 * @brief MindVision 构造函数
 * 逻辑：初始化参数，尝试首次开启，并建立守护进程监控相机在线状态。
 */
MindVision::MindVision(double exposure_ms, double gamma, const std::string & vid_pid)
: exposure_ms_(exposure_ms), gamma_(gamma), handle_(-1), quit_(false), ok_(false), queue_(1), vid_(-1), pid_(-1)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("libusb 初始化失败！");

  try_open();

  // 100ms 周期的守护循环
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(100ms);
      if (ok_) continue; // 抓图线程正常运行时跳过

      if (capture_thread_.joinable()) capture_thread_.join();
      
      close();
      reset_usb(); // 硬件重置
      try_open();
    }
  }};
}

/**
 * @brief 析构实现
 */
MindVision::~MindVision()
{
  quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
  close();
  tools::logger()->info("迈德威视相机已销毁。");
}

/**
 * @brief 异步读取接口
 */
void MindVision::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);
  img = data.img;
  timestamp = data.timestamp;
}

/**
 * @brief 核心开启与配置逻辑
 * 逻辑：
 * 1. 初始化 SDK (CameraSdkInit)。
 * 2. 枚举并初始化第一个发现的相机 (CameraInit)。
 * 3. 获取相机性能参数 (CameraGetCapability)。
 * 4. 置位手动曝光模式，加载用户参数。
 * 5. 启动图像处理引擎 (CameraPlay) 并启动回调取图线程。
 */
void MindVision::open()
{
  int camera_num = 1;
  tSdkCameraDevInfo camera_info_list;
  tSdkCameraCapbility camera_capbility;
  
  CameraSdkInit(1);
  if (CameraEnumerateDevice(&camera_info_list, &camera_num) != CAMERA_STATUS_SUCCESS || camera_num == 0)
    throw std::runtime_error("未找到迈德威视相机！");

  if (CameraInit(&camera_info_list, -1, -1, &handle_) != CAMERA_STATUS_SUCCESS)
    throw std::runtime_error("相机初始化 (CameraInit) 失败！");

  CameraGetCapability(handle_, &camera_capbility);
  width_ = camera_capbility.sResolutionRange.iWidthMax;
  height_ = camera_capbility.sResolutionRange.iHeightMax;

  // 硬件参数注入
  CameraSetAeState(handle_, FALSE);                        // 手动曝光模式
  CameraSetExposureTime(handle_, exposure_ms_ * 1e3);      // 设置曝光 (单位 us)
  CameraSetGamma(handle_, gamma_ * 1e2);                   // 设置 Gamma
  CameraSetIspOutFormat(handle_, CAMERA_MEDIA_TYPE_BGR8);  // SDK 内部直接转换好 BGR 格式
  CameraSetTriggerMode(handle_, 0);                        // 连续采集
  CameraSetFrameSpeed(handle_, 1);                         // 高速模式

  CameraPlay(handle_);

  // 异步取图 Work Thread
  capture_thread_ = std::thread{[this] {
    tSdkFrameHead head;
    BYTE * raw_buffer;
    ok_ = true;
    
    while (!quit_) {
      // 阻塞式获取图像缓冲区 (100ms 超时)
      if (CameraGetImageBuffer(handle_, &head, &raw_buffer, 100) == CAMERA_STATUS_SUCCESS) {
        auto ts = std::chrono::steady_clock::now();
        cv::Mat processed_img(height_, width_, CV_8UC3);
        
        // 调用 SDK 内置 ISP (色彩校正、锐化等)
        CameraImageProcess(handle_, raw_buffer, processed_img.data, &head);
        CameraReleaseImageBuffer(handle_, raw_buffer);

        queue_.push({processed_img, ts});
      } else {
        ok_ = false; // 掉线标识
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
  }};

  tools::logger()->info("迈德威视相机已开启。");
}

/**
 * @brief 硬件重启 USB 端口
 */
void MindVision::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;
  auto usb_handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!usb_handle) return;
  
  if (libusb_reset_device(usb_handle))
    tools::logger()->warn("迈德威视 USB 端口硬件重置失败！");
  else
    tools::logger()->info("迈德威视 USB 端口重置成功。");

  libusb_close(usb_handle);
}

}  // namespace io
