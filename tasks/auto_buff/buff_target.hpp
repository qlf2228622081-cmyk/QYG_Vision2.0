#ifndef AUTO_BUFF__TARGET_HPP
#define AUTO_BUFF__TARGET_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "buff_detector.hpp"
#include "buff_type.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/ransac_sine_fitter.hpp"

namespace auto_buff
{

/**
 * @brief 投票器 (Voter)
 * 逻辑：用于判断能量机关的旋转方向（顺时针或逆时针）。
 * 通过多帧角度累积投票，确保方向判定的稳定性。
 */
class Voter
{
public:
  Voter();
  /**
   * @brief 根据前后帧角度差进行投票
   */
  void vote(const double angle_last, const double angle_now);
  /**
   * @brief 获取当前判定的旋转方向 (1: 顺时针, -1: 逆时针)
   */
  int clockwise();

private:
  int clockwise_; // 累积投票计数值
};

/**
 * @brief 能量机关目标基类 (Target)
 * 定义了能量机关追踪与预测的通用接口和核心卡尔曼滤波框架。
 */
class Target
{
public:
  Target();
  
  /**
   * @brief 处理每一帧的观测数据
   * @param p 检测到的能量机关对象
   * @param timestamp 图像时间戳
   */
  virtual void get_target(
    const std::optional<PowerRune> & p,
    std::chrono::steady_clock::time_point & timestamp) = 0;

  /**
   * @brief 预测未来 dt 时间后的状态
   */
  virtual void predict(double dt) = 0;

  /**
   * @brief 坐标转换：将能量机关局部系下的点投影到世界坐标系
   * @param point_in_buff 局部 3D 点 (R 标为原点)
   * @return Eigen::Vector3d 世界系坐标
   */
  Eigen::Vector3d point_buff2world(const Eigen::Vector3d & point_in_buff) const;

  /**
   * @brief 判定目标是否处于无法计算的状态
   */
  bool is_unsolve() const;

  /**
   * @brief 获取卡尔曼滤波器的当前状态向量 x
   */
  Eigen::VectorXd ekf_x() const;

  double spd = 0;  // 调试辅助变量

protected:
  // 内部初始化与更新接口
  virtual void init(double nowtime, const PowerRune & p) = 0;
  virtual void update(double nowtime, const PowerRune & p) = 0;

  // EKF 配置参数
  Eigen::VectorXd x0_;
  Eigen::MatrixXd P0_;
  Eigen::MatrixXd A_;
  Eigen::MatrixXd Q_;
  Eigen::MatrixXd H_;
  Eigen::MatrixXd R_;
  tools::ExtendedKalmanFilter ekf_; // 扩展卡尔曼滤波器

  double lasttime_ = 0;             // 上一次更新的相对时间
  Voter voter;                      // 旋转方向判定器
  bool first_in_;                   // 状态初始化标志位
  bool unsolvable_;                 // 目标无效标志位
};

/**
 * @brief 小符目标子类 (SmallTarget)
 * 对应能量机关的恒定转速模式 (SMALL)，匀速圆周运动模型。
 */
class SmallTarget : public Target
{
public:
  SmallTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;

private:
  void init(double nowtime, const PowerRune & p) override;
  void update(double nowtime, const PowerRune & p) override;

  /**
   * @brief 计算观测方程的雅可比矩阵
   */
  Eigen::MatrixXd h_jacobian() const;

  const double SMALL_W = CV_PI / 3; // 小符标准转速 (60度/s)
};

/**
 * @brief 大符目标子类 (BigTarget)
 * 对应能量机关的变速转速模式 (BIG)，通常符合正弦规律变加速度圆周运动模型。
 */
class BigTarget : public Target
{
public:
  BigTarget();

  void get_target(
    const std::optional<PowerRune> & p, std::chrono::steady_clock::time_point & timestamp) override;

  void predict(double dt) override;

private:
  void init(double nowtime, const PowerRune & p) override;
  void update(double nowtime, const PowerRune & p) override;

  /**
   * @brief 计算观测方程的雅可比矩阵
   */
  Eigen::MatrixXd h_jacobian() const;

  tools::RansacSineFitter spd_fitter_; // 辅助正弦拟合器，用于处理速度波动剧烈的情况

  double fit_spd_; // 拟合出的当前理想速度
};

}  // namespace auto_buff
#endif