#ifndef AUTO_BUFF__TRACK_HPP
#define AUTO_BUFF__TRACK_HPP

#include <yaml-cpp/yaml.h>

#include <deque>
#include <optional>

#include "buff_type.hpp"
#include "tools/img_tools.hpp"
#include "yolo11_buff.hpp"

// 目标丢失最大允许帧数，超过后重置追踪状态
const int LOSE_MAX = 20;

namespace auto_buff
{

/**
 * @brief 能量机关检测器类 (Buff_Detector)
 * 逻辑：负责调用神经网络模型检测扇叶，并结合时序信息计算旋转中心（R 标），
 * 最终输出完整的 PowerRune 目标信息。
 */
class Buff_Detector
{
public:
  /**
   * @brief 构造函数
   * @param config 配置文件路径
   */
  Buff_Detector(const std::string & config);

  /**
   * @brief 多目标/多候选框检测模式
   * 适用于视野内有多个扇叶亮起的情况，通过多帧匹配找到最新的 Target。
   */
  std::optional<PowerRune> detect_24(cv::Mat & bgr_img);

  /**
   * @brief 单目标检测模式
   */
  std::optional<PowerRune> detect(cv::Mat & bgr_img);

  /**
   * @brief 调试用的检测接口
   */
  std::optional<PowerRune> detect_debug(cv::Mat & bgr_img, cv::Point2f v);

private:
  /**
   * @brief 图像预处理
   * 流程：灰度化 -> 二值化 -> 膨胀。主要用于辅助精细化定位 R 标。
   */
  void handle_img(const cv::Mat & bgr_img, cv::Mat & dilated_img);

  /**
   * @brief 计算旋转中心 (R 标)
   * 逻辑：首先基于回归出来的关键点方向推算 R 标大致位置，随后在局部区域内
   * 通过传统视觉方法（轮廓查找+最小外接矩形）定位 R 标的精确物理中心。
   */
  cv::Point2f get_r_center(std::vector<FanBlade> & fanblades, cv::Mat & bgr_img);

  /**
   * @brief 处理目标丢失逻辑
   */
  void handle_lose();

  YOLO11_BUFF MODE_;                      // YOLO11 能量机关模型
  Track_status status_;                   // 当前追踪状态
  int lose_;                              // 当前丢失帧计数
  double lastlen_;                        // 缓存的长度信息 (用于防抖)
  std::optional<PowerRune> last_powerrune_ = std::nullopt; // 上一帧解析出的能量机关状态
};
}  // namespace auto_buff

#endif  // DETECTOR_HPP