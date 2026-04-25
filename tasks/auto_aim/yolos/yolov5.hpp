#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

/**
 * @brief YOLOv5 推理器类
 * 实现基于 OpenVINO 的 YOLOv5 目标检测。支持 ROI 截取、关键点（角点）回归，
 * 并可选地结合传统几何 Detector 进行角点精细化。
 */
class YOLOV5 : public YOLOBase
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径
   * @param debug 是否开启调试窗口
   */
  YOLOV5(const std::string & config_path, bool debug);

  /**
   * @brief 执行完整检测流程 (预处理 + 推理 + 后处理)
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /**
   * @brief 仅执行后处理逻辑
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_; // 推理设备 (CPU/GPU/AUTO) 及模型路径
  std::string save_path_, debug_path_; // 图像保存路径及调试路径
  bool debug_, use_roi_, use_traditional_; // 模式开关

  // 模型输出常数
  const int class_num_ = 13;      // 类别总数
  const float nms_threshold_ = 0.3;  // 非极大值抑制门限
  const float score_threshold_ = 0.7; // 置信度得分门限
  double min_confidence_, binary_threshold_;

  ov::Core core_;                   // OpenVINO 核心
  ov::CompiledModel compiled_model_; // 编译后的可执行模型

  cv::Rect roi_;          // 感兴趣区域范围
  cv::Point2f offset_;    // ROI 带来的坐标偏移量
  cv::Mat tmp_img_;       // 临时图像缓存（用于保存调试图）

  Detector detector_;     // 传统检测器，用于角点二次校正
  friend class MultiThreadDetector;

  // 内部辅助校验函数
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  /**
   * @brief 获取归一化中心坐标
   */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /**
   * @brief 解析神经网络原始输出张量
   */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  /**
   * @brief 保存图像到本地（用于数据集迭代）
   */
  void save(const Armor & armor) const;

  /**
   * @brief 在图像上绘制识别结果
   */
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  /**
   * @brief Sigmoid 激活函数
   */
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP