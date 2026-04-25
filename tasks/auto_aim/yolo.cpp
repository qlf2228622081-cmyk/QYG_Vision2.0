#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolo11.hpp"
#include "yolos/yov5.hpp"
#include "yolos/yov8.hpp"

namespace auto_aim
{

/**
 * @brief YOLO 门面类构造函数
 * 根据配置文件中的 yolo_name 字段，利用简单工厂模式创建具体的模型实现。
 */
YOLO::YOLO(const std::string & config_path, bool debug)
{
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  // 1. 动态选择 YOLO 版本分支
  if (yolo_name == "yov8" || yolo_name == "yolov8") {
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
  }

  else if (yolo_name == "yolo11") {
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
  }

  else if (yolo_name == "yov5" || yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  // 2. 异常处理：不支持的版本
  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

/**
 * @brief 转发检测请求到具体实现
 */
std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

/**
 * @brief 转发后处理请求到具体实现
 */
std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim