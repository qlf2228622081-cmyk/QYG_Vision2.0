#ifndef AUTO_AIM__YOLOV8_HPP
#define AUTO_AIM__YOLOV8_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/classifier.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace auto_aim
{

/**
 * @brief YOLOv8 推理器类
 * 实现基于 OpenVINO 的 YOLOv8 目标检测。与 v5 不同，v8 模型通常只输出
 * 装甲板的存在性，具体的兵种分类通过独立的 Classifier 模型完成。
 */
class YOLOV8 : public YOLOBase
{
public:
  /**
   * @brief 构造函数
   */
  YOLOV8(const std::string & config_path, bool debug);

  /**
   * @brief 执行检测流水线
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /**
   * @brief 后处理逻辑接口
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  Classifier classifier_; // 独立的分类器，用于识别数字/兵种
  Detector detector_;     // 传统检测器，辅助角点校正（可选）

  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  const int class_num_ = 2; // YOLOv8 训练时的类别数
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_;

  // 内部校验逻辑
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  /**
   * @brief 提取装甲板中心图案，供分类器识别
   */
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;
  
  /**
   * @brief 根据识别出的兵种名称推断装甲板类型 (大/小)
   */
  ArmorType get_type(const Armor & armor);
  
  /**
   * @brief 坐标归一化
   */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /**
   * @brief 解析 YOLOv8 输出张量
   */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  
  /**
   * @brief 角点排序函数
   * 确保回归出的 4 个角点按照 (TL, TR, BR, BL) 的固定顺序排列，以便后续 PnP。
   */
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  // TOOLS__YOLOV8_HPP