#include "hikrobot.cpp"

#include <libusb-1.0/libusb.h>
#include "tools/logger.hpp"

using namespace std::chrono_literals;

namespace io
{

/**
 * @brief HikRobot 构造函数 (USB 版本)
 * 逻辑：启动守护线程。守护进程会始终循环检查 capturing_ 标志，
 * 若相机掉线或抓图异常，会尝试通过 libusb 重置 USB 端口并重新枚举设备。
 */
HikRobot::HikRobot(double exposure_ms, double gain, const std::string & vid_pid)
: exposure_us_(exposure_ms * 1e3), gain_(gain), queue_(1), daemon_quit_(false), vid_(-1), pid_(-1), handle_(nullptr)
{
  set_vid_pid(vid_pid);
  if (libusb_init(NULL)) tools::logger()->warn("libusb 初始化失败！");

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("海康相机守护线程已启动。");
    capture_start(); // 初始开启

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);
      if (capturing_) continue; // 只要取图线程工作正常就跳过

      tools::logger()->warn("检测到相连接断开，正在尝试硬件重置与重连...");
      capture_stop();
      reset_usb(); // 硬件重置端口
      capture_start();
      
      // 等待几帧确保重连成功
      for (int i = 0; i < 5 && !capturing_; ++i) std::this_thread::sleep_for(100ms);
    }
  }};
}

/**
 * @brief 接收数据接口
 * 逻辑：从内部 ThreadSafeQueue 中取出由 capture_thread 填充的最新图像。
 */
void HikRobot::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data);
  img = data.img;
  timestamp = data.timestamp;
}

/**
 * @brief 核心抓图与配置逻辑 (USB 版)
 * 步骤：
 * 1. 遍历 MV_USB_DEVICE 列表。
 * 2. 创建设备句柄 (CreateHandle) 并打开设备 (OpenDevice)。
 * 3. 强制关闭自动曝光/增益，设置手动参数，配置白平衡。
 * 4. 开启抓图 (StartGrabbing) 并启动消费者线程。
 */
void HikRobot::capture_start()
{
  unsigned int ret;
  MV_CC_DEVICE_INFO_LIST device_list;
  
  // 1. 获取设备列表
  ret = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (ret != MV_OK || device_list.nDeviceNum == 0) {
    handle_ = nullptr; return;
  }

  // 2. 初始化句柄
  ret = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[0]);
  if (ret != MV_OK || MV_CC_OpenDevice(handle_) != MV_OK) {
    handle_ = nullptr; return;
  }

  // 3. 配置相机寄存器
  set_enum_value("BalanceWhiteAuto", MV_BALANCEWHITE_AUTO_CONTINUOUS);
  set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
  set_enum_value("GainAuto", MV_GAIN_MODE_OFF);
  set_enum_value("TriggerMode", MV_TRIGGER_MODE_OFF);
  set_float_value("ExposureTime", exposure_us_);
  set_float_value("Gain", gain_);
  MV_CC_SetFrameRate(handle_, 150); // 期望帧率 400fps 以上，视曝光而定

  // 4. 开启 SDK 内部抓图
  if (MV_CC_StartGrabbing(handle_) != MV_OK) {
    capture_stop(); return;
  }

  // 5. 启动图像解析子线程
  capture_thread_ = std::thread{[this] {
    capturing_ = true;
    MV_FRAME_OUT raw;

    while (!capture_quit_) {
      // 从 SDK 缓冲区获取图像
      if (MV_CC_GetImageBuffer(handle_, &raw, 1000) != MV_OK) break;
      
      auto ts = std::chrono::steady_clock::now();
      
      // 这里的原始数据通常是 Bayer 格式，需要转换为 RGB
      cv::Mat bayer_img(cv::Size(raw.stFrameInfo.nWidth, raw.stFrameInfo.nHeight), CV_8U, raw.pBufAddr);
      cv::Mat rgb_img;
      
      // 根据海康像素格式选择 OpenCV 转换类型
      auto pixel_type = raw.stFrameInfo.enPixelType;
      const static std::unordered_map<MvGvspPixelType, cv::ColorConversionCodes> type_map = {
        {PixelType_Gvsp_BayerGR8, cv::COLOR_BayerGR2RGB},
        {PixelType_Gvsp_BayerRG8, cv::COLOR_BayerRG2RGB},
        {PixelType_Gvsp_BayerGB8, cv::COLOR_BayerGB2RGB},
        {PixelType_Gvsp_BayerBG8, cv::COLOR_BayerBG2RGB}};
      
      try {
        cv::cvtColor(bayer_img, rgb_img, type_map.at(pixel_type));
        queue_.push({rgb_img, ts});
      } catch(...) {
        tools::logger()->error("不支持的像素格式转换！");
      }

      MV_CC_FreeImageBuffer(handle_, &raw);
    }
    capturing_ = false;
  }};
}

/**
 * @brief 硬件重启 USB 端口
 * 逻辑：针对工业相机在复杂干扰下锁死无法通过软件 Reset 的情况，
 * 调用 libusb 模拟物理插拔动作。
 */
void HikRobot::reset_usb() const
{
  if (vid_ == -1 || pid_ == -1) return;
  auto handle = libusb_open_device_with_vid_pid(NULL, vid_, pid_);
  if (!handle) return;
  
  if (libusb_reset_device(handle))
    tools::logger()->warn("USB 硬件重置失败！");
  else
    tools::logger()->info("USB 端口重置成功。");

  libusb_close(handle);
}

}  // namespace io