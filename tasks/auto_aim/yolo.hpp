#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief YOLO 抽象基类 (Interface)
 * 定义了不同版本 YOLO (v5, v8, v11) 必须实现的通用检测接口。
 */
class YOLOBase
{
public:
  /**
   * @brief 纯虚检测函数
   * @param img 输入待检测图像
   * @param frame_count 帧序列号 (用于缓存或调试)
   * @return std::list<Armor> 检测到的装甲板列表
   */
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  /**
   * @brief 纯虚后处理函数
   * 将神经网络输出张量解析为 Armor 对象，包括坐标缩放、NMS 等。
   */
  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

/**
 * @brief YOLO 门面类 (Facade/Wrapper)
 * 通过配置文件动态实例化具体版本的 YOLO 探测器，对外部提供统一 API。
 */
class YOLO
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式（显示检测框等）
   */
  YOLO(const std::string & config_path, bool debug = true);

  /**
   * @brief 执行检测逻辑
   */
  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  /**
   * @brief 执行后处理逻辑
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_; // 指向具体实现的基类指针
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP