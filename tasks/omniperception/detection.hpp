#ifndef OMNIPERCEPTION__DETECTION_HPP
#define OMNIPERCEPTION__DETECTION_HPP

#include <chrono>
#include <list>

#include "tasks/auto_aim/armor.hpp"

namespace omniperception
{

/**
 * @brief 全场感知识别结果结构体 (DetectionResult)
 * 一个识别结果封装了单个相机在某一时刻探测到的所有装甲板信息，
 * 以及该相机相对于当前云台朝向的角度偏差。
 */
struct DetectionResult
{
  std::list<auto_aim::Armor> armors;               // 该相机视野内的装甲板列表
  std::chrono::steady_clock::time_point timestamp; // 图像采集时间戳
  double delta_yaw;                                // 相机中心相对于目标的 Yaw 偏差 (rad)
  double delta_pitch;                              // 相机中心相对于目标的 Pitch 偏差 (rad)

  /**
   * @brief 赋值运算符重载，用于线程安全队列中的数据拷贝
   */
  DetectionResult & operator=(const DetectionResult & other)
  {
    if (this != &other) {
      armors = other.armors;
      timestamp = other.timestamp;
      delta_yaw = other.delta_yaw;
      delta_pitch = other.delta_pitch;
    }
    return *this;
  }
};
}  // namespace omniperception

#endif