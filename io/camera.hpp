#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

namespace io
{

/**
 * @brief 工业相机抽象基类 (CameraBase)
 * 逻辑：定义了访问工业相机的通用接口。不同的 SDK（如海康 HikRobot、迈德威视 MindVision）
 * 通过实现该基类，向上层视觉算法提供统一的取图方式。
 */
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  
  /**
   * @brief 读取一帧图像及其对应的硬件/系统时间戳
   */
  virtual void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) = 0;
};

/**
 * @brief 工业相机管理器 (Camera)
 * 逻辑：作为一个“工厂类”包装器，根据配置文件中的 camera_name 动态实例化特定的相机驱动。
 * 支持海康 (HikRobot) 和迈德威视 (MindVision) 两种主要的工业级相机。
 */
class Camera
{
public:
  /**
   * @brief 构造函数：解析配置文件并初始化驱动
   */
  Camera(const std::string & config_path);

  /**
   * @brief 读取接口：内部重定向到具体的 SDK 实现
   */
  void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);

private:
  std::unique_ptr<CameraBase> camera_; // 具体相机驱动实例的唯一指针
};

}  // namespace io

#endif  // IO__CAMERA_HPP