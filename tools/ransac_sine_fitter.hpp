#pragma once

#include <Eigen/Dense>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

namespace tools
{

/**
 * @brief RANSAC 正弦拟合器 (RansacSineFitter)
 * 逻辑：利用 RANSAC 算法对变加速度圆周运动中的角速度（正弦规律变化）进行拟合。
 * 能够有效剔除观测噪声及野值，精准提取正弦函数的振幅 (A)、频率 (omega)、相位 (phi) 和偏置 (C)。
 * 公式：v = A * sin(omega * t + phi) + C
 */
class RansacSineFitter
{
public:
  /**
   * @brief 拟合结果结构体
   */
  struct Result
  {
    double A = 0.0;     // 振幅
    double omega = 0.0; // 角频率
    double phi = 0.0;   // 初相位
    double C = 0.0;     // 偏置 (即基准速度)
    int inliers = 0;    // 内点计数
  };
  Result best_result_; // 保存全局最优拟合结果

  /**
   * @brief 构造函数
   * @param max_iterations 最大 RANSAC 迭代次数
   * @param threshold 内点阈值误差
   * @param min_omega/max_omega 频率搜索区间 (依据 Robomaster 规则设定)
   */
  RansacSineFitter(int max_iterations, double threshold, double min_omega, double max_omega);

  /**
   * @brief 添加观测数据对 (时间 t, 速度 v)
   */
  void add_data(double t, double v);

  /**
   * @brief 执行拟合计算
   */
  void fit();

  /**
   * @brief 理想正弦函数值计算接口
   */
  double sine_function(double t, double A, double omega, double phi, double C)
  {
    return A * std::sin(omega * t + phi) + C;
  }

private:
  int max_iterations_;
  double threshold_;
  double min_omega_;
  double max_omega_;
  std::mt19937 gen_;                            // 随机数引擎
  std::deque<std::pair<double, double>> fit_data_; // 历史观测数据滑窗

  /**
   * @brief 基于最小二乘求解部分模型参数 (A, phi, C)
   * 在预假设 omega 的前提下，由于模型变为线性，可利用 SVD 快速求解。
   */
  bool fit_partial_model(
    const std::vector<std::pair<double, double>> & sample, double omega, Eigen::Vector3d & params);

  /**
   * @brief 统计当前模型符合阈值要求的观测点数量
   */
  int evaluate_inliers(double A, double omega, double phi, double C);
};

}  // namespace tools
