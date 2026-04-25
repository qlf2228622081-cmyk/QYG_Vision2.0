#include "usbcamera.cpp"

#include <stdexcept>
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace io
{

/**
 * @brief USBCamera 构造函数
 * 逻辑：读取配置并启动“守护线程”，守护线程负责相机的生命周期。
 */
USBCamera::USBCamera(const std::string & open_name, const std::string & config_path)
: open_name_(open_name), quit_(false), ok_(false), queue_(1), open_count_(0)
{
  auto yaml = tools::load(config_path);
  image_width_ = tools::read<double>(yaml, "image_width");
  image_height_ = tools::read<double>(yaml, "image_height");
  usb_exposure_ = tools::read<double>(yaml, "usb_exposure");
  usb_frame_rate_ = tools::read<double>(yaml, "usb_frame_rate");
  usb_gamma_ = tools::read<double>(yaml, "usb_gamma");
  usb_gain_ = tools::read<double>(yaml, "usb_gain");

  // 初次尝试开启相机
  try_open();

  // 为相机配置守护后台：
  // 1. 监控健康状态，若掉线则定期尝试重开。
  // 2. 防止一次启动失败导致的程序崩溃。
  daemon_thread_ = std::thread{[this] {
    while (!quit_) {
      std::this_thread::sleep_for(100ms);
      if (ok_) continue; // 状态健康，跳过

      // 如果重试次数过多 (20次，约2秒)，则停止尝试
      if (open_count_ > 20) {
        tools::logger()->warn("无法开启 {} USB 相机，已停止重试。", this->device_name);
        quit_ = true;
        close();
        break;
      }

      // 重启逻辑：清理旧线程后重新 open
      if (capture_thread_.joinable()) capture_thread_.join();
      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        close();
      }
      try_open();
    }
  }};
}

/**
 * @brief 析构实现
 */
USBCamera::~USBCamera()
{
  quit_ = true;
  {
    std::lock_guard<std::mutex> lock(cap_mutex_);
    close();
  }
  if (daemon_thread_.joinable()) daemon_thread_.join();
  if (capture_thread_.joinable()) capture_thread_.join();
  tools::logger()->info("USBCamera 设备已安全释放。");
}

/**
 * @brief 阻塞读取一帧原始数据 (通常由内部子线程调用)
 */
cv::Mat USBCamera::read()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  if (!cap_.isOpened()) return cv::Mat();
  cap_ >> img_;
  return img_;
}

/**
 * @brief 外部读取接口：从异步队列中弹出一帧数据
 */
void USBCamera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  CameraData data;
  queue_.pop(data); // 阻塞直到队列中有新帧

  img = data.img;
  timestamp = data.timestamp;
}

/**
 * @brief 底层开启相机并启动取图任务
 * 逻辑：基于 V4L 协议开启相机，设置 MJPG 编码格式、分辨率、曝光等关键硬件参数。
 */
void USBCamera::open()
{
  std::lock_guard<std::mutex> lock(cap_mutex_);
  std::string true_device_name = "/dev/" + open_name_;
  cap_.open(true_device_name, cv::CAP_V4L);
  
  if (!cap_.isOpened()) {
    tools::logger()->warn("OpenCV 无法打开设备 {}", true_device_name);
    return;
  }

  // 这里的逻辑利用了“锐度 (Sharpness)”寄存器来区分是左相机还是右相机 (电控预埋配置)
  sharpness_ = cap_.get(cv::CAP_PROP_SHARPNESS);
  
  // 设置高帧率 MJPG 模式
  cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_.set(cv::CAP_PROP_FPS, usb_frame_rate_);
  cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1); // 1 通常代表手动曝光模型
  cap_.set(cv::CAP_PROP_EXPOSURE, usb_exposure_);
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
  cap_.set(cv::CAP_PROP_GAMMA, usb_gamma_);
  cap_.set(cv::CAP_PROP_GAIN, usb_gain_);

  // 根据预设锐度自动赋予设备标签
  if (sharpness_ == 2) device_name = "left";
  else if (sharpness_ == 3) device_name = "right";
  else device_name = "unknown";

  tools::logger()->info("{} USB 相机已开启，期望帧率: {}", device_name, usb_frame_rate_);

  // 启动异步取图子线程 (Worker)
  capture_thread_ = std::thread{[this] {
    ok_ = true;
    tools::logger()->info("[{} USB camera] 取图子线程已启动。", this->device_name);
    while (!quit_) {
      cv::Mat temp_img;
      bool success;
      {
        std::lock_guard<std::mutex> lock(cap_mutex_);
        if (!cap_.isOpened()) break;
        success = cap_.read(temp_img); // 执行硬件取图
      }

      if (!success) {
        tools::logger()->warn("读取帧失败，取图线程即将退出。");
        break;
      }

      // 记录时间戳并压入结果队列
      auto ts = std::chrono::steady_clock::now();
      queue_.push({temp_img, ts});
      
      std::this_thread::sleep_for(1ms);
    }
    ok_ = false;
  }};
}

void USBCamera::try_open()
{
  try {
    open();
    open_count_++;
  } catch (const std::exception & e) {
    tools::logger()->warn("USBCamera open error: {}", e.what());
  }
}

void USBCamera::close()
{
  if (cap_.isOpened()) {
    cap_.release();
    tools::logger()->info("USB 相机资源已释放。");
  }
}

}  // namespace io