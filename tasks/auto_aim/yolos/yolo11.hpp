#ifndef AUTO_AIM__YOLO11_HPP
#define AUTO_AIM__YOLO11_HPP

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
 * @brief YOLO11 推理器类
 * 实现基于 OpenVINO 的 YOLO11 目标检测。相比之前的版本，YOLO11 在检测精度和速度上有更好的平衡。
 * 支持 4 个关键点（角点）回归及多种兵种分类的端到端识别（不一定依赖外部 Classifier）。
 */
class YOLO11 : public YOLOBase
{
public:
  /**
   * @brief 构造函数
   */
  YOLO11(const std::string & config_path, bool debug);

  /**
   * @brief 执行检测流水线
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /**
   * @brief 后处理接口
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_;

  const int class_num_ = 38; // 预设的分类数（可能包含颜色和兵种的组合）
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  Detector detector_; // 传统几何检测器（辅助校正）

  // 内部校验
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  /**
   * @brief 归一化中心点
   */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /**
   * @brief 解析 YOLO11 原始输出
   */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  
  /**
   * @brief 角点排序
   */
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLO11_HPP