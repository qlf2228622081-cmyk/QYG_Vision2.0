#include "ransac_sine_fitter.cpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

namespace tools
{

/**
 * @brief RansacSineFitter 构造函数
 */
RansacSineFitter::RansacSineFitter(
  int max_iterations, double threshold, double min_omega, double max_omega)
: max_iterations_(max_iterations),
  threshold_(threshold),
  min_omega_(min_omega),
  max_omega_(max_omega),
  gen_(std::random_device{}())
{
}

/**
 * @brief 添加数据点
 * 逻辑：如果时间戳跳变过大（>5s），则重置缓存，防止旧数据干扰拟合。
 */
void RansacSineFitter::add_data(double t, double v)
{
  if (fit_data_.size() > 0 && (t - fit_data_.back().first > 5)) fit_data_.clear();
  fit_data_.emplace_back(std::make_pair(t, v));
}

/**
 * @brief RANSAC 主干逻辑实现
 * 1. 随机选取 3 个样本点。
 * 2. 在指定频率区间内随机采样一个 omega。
 * 3. 利用线性最小二乘推导 A1*sin(wt) + A2*cos(wt) + C 的系数。
 * 4. 统计内点数，记录全局最佳参数。
 */
void RansacSineFitter::fit()
{
  if (fit_data_.size() < 3) return;

  std::uniform_real_distribution<double> omega_dist(min_omega_, max_omega_);
  std::vector<size_t> indices(fit_data_.size());
  std::iota(indices.begin(), indices.end(), 0);

  best_result_.inliers = 0; // 重置本轮搜索结果

  for (int iter = 0; iter < max_iterations_; ++iter) {
    // 随机洗牌样本索引
    std::shuffle(indices.begin(), indices.end(), gen_);

    std::vector<std::pair<double, double>> sample;
    for (int i = 0; i < 3; ++i) {
      sample.push_back(fit_data_[indices[i]]);
    }

    // 假设频率 omega
    double omega = omega_dist(gen_);
    Eigen::Vector3d params;
    
    // 如果该 omega 下无法解出模型，跳过
    if (!fit_partial_model(sample, omega, params)) continue;

    // 模型展开式：A1*sin(wt) + A2*cos(wt) + C
    double A1 = params(0); double A2 = params(1); double C = params(2);

    // 换算回标准幅相形式：A*sin(wt + phi) + C
    double A = std::sqrt(A1 * A1 + A2 * A2);
    double phi = std::atan2(A2, A1);

    // 统计该参数下的符合度
    int inlier_count = evaluate_inliers(A, omega, phi, C);

    if (inlier_count > best_result_.inliers) {
      best_result_.A = A;
      best_result_.omega = omega;
      best_result_.phi = phi;
      best_result_.C = C;
      best_result_.inliers = inlier_count;
    }
  }

  // 滑窗维护：保持数据点数量在合理范围内 (150帧)，确保实时性
  if (fit_data_.size() > 150) fit_data_.pop_front();
}

/**
 * @brief 线性最小二乘求解子模型 (基于给定的 omega)
 * 设计：将 v = A1*sin(wt) + A2*cos(wt) + C 对 A1, A2, C 线性化。
 * 求解超定方程组 Xβ = Y。
 */
bool RansacSineFitter::fit_partial_model(
  const std::vector<std::pair<double, double>> & sample, double omega, Eigen::Vector3d & params)
{
  Eigen::MatrixXd X(sample.size(), 3);
  Eigen::VectorXd Y(sample.size());

  for (size_t i = 0; i < sample.size(); ++i) {
    double t = sample[i].first;
    double y = sample[i].second;
    X(i, 0) = std::sin(omega * t);
    X(i, 1) = std::cos(omega * t);
    X(i, 2) = 1.0;
    Y(i) = y;
  }

  try {
    // 利用 SVD (Divide and Conquer) 求解线性方程组
    params = X.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Y);
    return true;
  } catch (...) {
    return false;
  }
}

/**
 * @brief 内点统计
 */
int RansacSineFitter::evaluate_inliers(double A, double omega, double phi, double C)
{
  int count = 0;
  for (const auto & p : fit_data_) {
    double pred = A * std::sin(omega * p.first + phi) + C;
    if (std::abs(p.second - pred) < threshold_) ++count;
  }
  return count;
}

}  // namespace tools
