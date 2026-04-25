#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor.hpp"
#include "classifier.hpp"

namespace auto_aim
{

/**
 * @brief 装甲板检测类 (Detector)
 * 采用传统几何特征过滤的方法识别装甲板，包含了灯条提取、装甲板配对及分类。
 */
class Detector
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径，包含各项几何约束阈值
   * @param debug 是否开启调试模式（显示中间图像）
   */
  Detector(const std::string & config_path, bool debug = true);

  /**
   * @brief 全图检测装甲板
   * @param bgr_img 输入的 BGR 图像
   * @param frame_count 当前帧序号，用于调试显示
   * @return std::list<Armor> 检测到的装甲板列表
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

  /**
   * @brief 局部搜索/ROI检测 (基于已有装甲板进行微调)
   * @param armor 输入并输出更新后的装甲板对象
   * @param bgr_img 输入的 BGR 图像
   * @return bool 是否检测成功
   */
  bool detect(Armor & armor, const cv::Mat & bgr_img);

  friend class YOLOV8; // 允许 YOLO 访问私有成员进行后处理适配

private:
  Classifier classifier_; // 装甲板数字分类器

  // 几何过滤阈值
  double threshold_;               // 灰度二值化阈值
  double max_angle_error_;         // 灯条最大倾斜角度误差
  double min_lightbar_ratio_, max_lightbar_ratio_; // 灯条长宽比范围
  double min_lightbar_length_;     // 灯条最小长度
  double min_armor_ratio_, max_armor_ratio_;       // 装甲板长宽比范围
  double max_side_ratio_;          // 装甲板两侧灯条长度比例差上限
  double min_confidence_;          // 分类器最低置信度
  double max_rectangular_error_;   // 矩形度误差上限

  bool debug_;           // 调试开关
  std::string save_path_; // 用于保存分类样本的路径

  /**
   * @brief 利用 PCA (主成分分析) 进行灯条角点回归
   * 提高强曝光或远距离下灯条端点的定位精度。
   * 参考自：CSU-FYT-Vision 方案
   */
  void lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const;

  // 几何逻辑检查函数
  bool check_geometry(const Lightbar & lightbar) const; // 检查单个灯条是否符合标准
  bool check_geometry(const Armor & armor) const;       // 检查配对后的装甲板是否符合标准
  bool check_name(const Armor & armor) const;           // 根据分类结果过滤（如空装甲）
  bool check_type(const Armor & armor) const;           // 检查大小装甲板逻辑一致性

  // 特征提取辅助函数
  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const; // 提取装甲板中心图案用于分类
  ArmorType get_type(const Armor & armor);                                // 自动判定大小装甲板
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const; // 归一化中心坐标

  // 调试辅助
  void save(const Armor & armor) const; // 保存装甲板截图用于训练数据集
  void show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
    const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__DETECTOR_HPP