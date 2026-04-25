#ifndef BUFF__TYPE_HPP
#define BUFF__TYPE_HPP

#include <algorithm>
#include <deque>
#include <eigen3/Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "tools/math_tools.hpp"

namespace auto_buff
{

const int INF = 1000000;

/**
 * @brief 能量机关类型枚举: 小符 / 大符
 */
enum PowerRune_type { SMALL, BIG };

/**
 * @brief 扇叶状态枚举: 目标扇叶(正在激活中) / 未激活扇叶 / 已激活扇叶
 */
enum FanBlade_type { _target, _unlight, _light };

/**
 * @brief 追踪状态机枚举
 */
enum Track_status { TRACK, TEM_LOSE, LOSE };

/**
 * @brief 扇叶类 (FanBlade)
 * 封装能量机关单个叶片的几何特征与状态。
 */
class FanBlade
{
public:
  cv::Point2f center;               // 扇叶数字区域的几何中心
  std::vector<cv::Point2f> points;  // 关键点列表 (通常 4 个角点，从左上角顺时针或特定顺序)
  double angle, width, height;      // 扇叶的倾角、物理宽度与高度
  FanBlade_type type;               // 当前状态 (目标/未激活/已激活)

  explicit FanBlade() = default;

  /**
   * @brief 构造函数
   * @param kpt 关键点坐标列表
   * @param keypoints_center 回归的中心点
   * @param t 初始状态类型
   */
  explicit FanBlade(
    const std::vector<cv::Point2f> & kpt, cv::Point2f keypoints_center, FanBlade_type t);

  explicit FanBlade(FanBlade_type t);
};

/**
 * @brief 能量机关整体类 (PowerRune)
 * 封装了 5 片扇叶的组合关系及其在 3D 空间中的坐标表达。
 */
class PowerRune
{
public:
  cv::Point2f r_center;             // 能量机关旋转中心 (R 标)
  std::vector<FanBlade> fanblades;  // 5 片扇叶列表 (内部按照以 Target 为起点顺时针排序)

  int light_num;                    // 当前已亮起的扇叶数量

  // 世界坐标系 (IMU 系) 下的位姿信息
  Eigen::Vector3d xyz_in_world;     // R 标位置 (x, y, z)
  Eigen::Vector3d ypr_in_world;     // R 标姿态 (Yaw, Pitch, Roll)
  Eigen::Vector3d ypd_in_world;     // R 标球坐标 (Yaw, Pitch, Distance)

  Eigen::Vector3d blade_xyz_in_world;  // 当前观测到的物理扇叶 (Target) 的 3D 位置
  Eigen::Vector3d blade_ypd_in_world;  // 物理扇叶的球坐标

  /**
   * @brief 构造函数
   * 包含核心逻辑：基于历史帧状态识别哪一片是最新亮起的 Target 扇叶。
   */
  explicit PowerRune(
    std::vector<FanBlade> & ts, const cv::Point2f r_center,
    std::optional<PowerRune> last_powerrune);
    
  explicit PowerRune() = default;

  /**
   * @brief 获取当前正待击打的预测目标扇叶
   */
  FanBlade & target() { return fanblades[0]; };

  /**
   * @brief 是否检测超时或由于几何错误失效
   */
  bool is_unsolve() const { return unsolvable_; }

private:
  double target_angle_;     // 目标相对于 R 标的角度
  bool unsolvable_ = false; // 是否为无效目标

  /**
   * @brief 计算点相对于 r_center 的极坐标角度 (0 to 2PI)
   */
  double atan_angle(cv::Point2f v) const;
};
}  // namespace auto_buff

#endif  // BUFF_TYPE_HPP
