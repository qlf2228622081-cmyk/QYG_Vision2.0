#ifndef TOOLS__EXTENDED_KALMAN_FILTER_HPP
#define TOOLS__EXTENDED_KALMAN_FILTER_HPP

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>

namespace tools
{

/**
 * @brief 扩展卡尔曼滤波器 (EKF)
 * 逻辑：用于处理非线性系统的状态估计。支持通过函数指针（std::function）定义非线性
 * 预测模型 f(x) 和非线性观测模型 h(x)。
 * 包含残差计算、卡方检验（NIS/NEES）等诊断工具。
 */
class ExtendedKalmanFilter
{
public:
  Eigen::VectorXd x; // 系统状态向量
  Eigen::MatrixXd P; // 状态协方差矩阵

  ExtendedKalmanFilter() = default;

  /**
   * @brief 构造函数
   * @param x0 初始状态
   * @param P0 初始协方差
   * @param x_add 状态加法逻辑 (用于处理角度循环等特殊域)
   */
  ExtendedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  /**
   * @brief 线性预测阶段：x = F*x, P = F*P*FT + Q
   */
  Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  /**
   * @brief 非线性预测阶段：x = f(x), P = F*P*FT + Q
   * @param F 状态转移矩阵的雅可比矩阵
   */
  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  /**
   * @brief 线性更新阶段
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  /**
   * @brief 非线性更新阶段：利用 h(x) 计算残差
   * @param H 观测方程的雅可比矩阵
   */
  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a - b; });

  std::map<std::string, double> data;  // 存储调试与性能分析数据 (如残差、卡方结果)
  std::deque<int> recent_nis_failures{0}; // 记录最近一段时间内的 NIS 失败情况 (用于发散检测)
  size_t window_size = 100;
  double last_nis;

private:
  Eigen::MatrixXd I; // 单位矩阵 (辅助计算)
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

  // 内部统计计数器
  int nees_count_ = 0;
  int nis_count_ = 0;
  int total_count_ = 0;
};

}  // namespace tools

#endif  // TOOLS__EXTENDED_KALMAN_FILTER_HPP