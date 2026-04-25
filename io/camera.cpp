#include "camera.cpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{

/**
 * @brief Camera 构造函数实现 (工厂模式)
 * 逻辑：
 * 1. 读取 YAML 中的相机品牌 (camera_name)。
 * 2. 依次读取曝光、增益、VID/PID 等通用或厂商专属参数。
 * 3. 实例化对应的 SDK 封装类并注入 CameraBase 指针。
 */
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");

  // 1. 迈德威视模式
  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  // 2. 海康 USB 模式
  else if (camera_name == "hikrobot") {
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
  }

  // 3. 海康 GigE (网口) 模式
  else if(camera_name == "hikrobot_gige") {
    auto gain = tools::read<double>(yaml, "gain");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain); // 网口版不需要 VID/PID
  }

  else {
    throw std::runtime_error("未知的相机品牌配置: " + camera_name + "，请检查 YAML 文件！");
  }
}

/**
 * @brief 转发取图请求
 */
void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  if (camera_) {
    camera_->read(img, timestamp);
  }
}

}  // namespace io