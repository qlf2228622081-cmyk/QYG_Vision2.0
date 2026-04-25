#ifndef AUTO_AIM__CLASSIFIER_HPP
#define AUTO_AIM__CLASSIFIER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief 装甲板数字分类器 (Classifier)
 * 核心逻辑：对检测到的装甲板中心区域（Pattern）进行处理，
 * 利用卷积神经网络（CNN）识别其所属兵种编号（1-5, 哨兵等）。
 * 支持 OpenCV DNN 和 OpenVINO 两种推理后端。
 */
class Classifier
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，包含模型文件路径
   */
  explicit Classifier(const std::string & config_path);

  /**
   * @brief 执行分类 (OpenCV DNN 后端)
   */
  void classify(Armor & armor);

  /**
   * @brief 执行分类 (OpenVINO 后端 - 通常推荐在 Intel CPU 上使用)
   */
  void ovclassify(Armor & armor);

private:
  cv::dnn::Net net_;                 // OpenCV DNN 网络实例
  ov::Core core_;                    // OpenVINO 核心对象
  ov::CompiledModel compiled_model_; // OpenVINO 编译后的模型
};

}  // namespace auto_aim

#endif  // AUTO_AIM__CLASSIFIER_HPP