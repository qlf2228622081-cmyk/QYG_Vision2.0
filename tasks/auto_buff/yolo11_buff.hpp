#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "tools/logger.hpp"

namespace auto_buff
{

// 能量机关识别类别名称: 扇叶(buff), R标(r)
const std::vector<std::string> class_names = {"buff", "r"};

/**
 * @brief YOLO11 能量机关专有检测器
 * 逻辑：基于 OpenVINO 的 YOLO11 关键点回归模型，专用于检测能量机关的扇叶角点及旋转中心。
 * 该模型通常输出矩形框、置信度以及 6 个关键点（4角点+中心+方向点）。
 */
class YOLO11_BUFF
{
public:
  /**
   * @brief 检测结果结构体
   */
  struct Object
  {
    cv::Rect_<float> rect;        // 目标检测框
    int label;                    // 类别索引
    float prob;                   // 置信度
    std::vector<cv::Point2f> kpt; // 关键点列表 (通常为 6 个)
  };

  /**
   * @brief 构造函数
   * @param config 包含模型路径和基本参数的配置文件路径
   */
  YOLO11_BUFF(const std::string & config);

  /**
   * @brief 获取所有符合条件的候选目标 (带 NMS)
   * 适用于视野内有多个已激活扇叶的情况。
   */
  std::vector<Object> get_multicandidateboxes(cv::Mat & image);

  /**
   * @brief 获取置信度最高的一个候选目标
   * 适用于快速锁定当前最显眼的扇叶。
   */
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  ov::Core core;                // OpenVINO 运行时核心
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiled_model;
  ov::InferRequest infer_request;
  ov::Tensor input_tensor;

  const int NUM_POINTS = 6;     // 关键点数量 (4角点+中心+方向参考)

  // 图像预处理与转换工具函数
  void convert(
    const cv::Mat & input, cv::Mat & output, const bool normalize, const bool exchangeRB) const;

  /**
   * @brief 填充输入张量
   * 包含 Letterbox 变换逻辑，保持长宽比缩放。
   */
  float fill_tensor_data_image(ov::Tensor & input_tensor, const cv::Mat & input_image) const;

  // 调试助手
  void printInputAndOutputsInfo(const ov::Model & network);
  void save(const std::string & programName, const cv::Mat & image);
};
}  // namespace auto_buff

#endif