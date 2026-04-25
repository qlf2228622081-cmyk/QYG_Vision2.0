#include "extended_kalman_filter.cpp"

#include <numeric>

namespace tools
{

/**
 * @brief EKF 初始化实现
 * 设置初始状态、协方差，并初始化调试记录表。
 */
ExtendedKalmanFilter::ExtendedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  // 零填充调试指标
  data["residual_yaw"] = 0.0; data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0; data["residual_angle"] = 0.0;
  data["nis"] = 0.0; data["nees"] = 0.0;
}

/**
 * @brief 线性预测
 */
Eigen::VectorXd ExtendedKalmanFilter::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

/**
 * @brief 预测阶段实现
 * 逻辑：xp = f(x), Pp = F*P*FT + Q
 */
Eigen::VectorXd ExtendedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

/**
 * @brief 线性更新
 */
Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

/**
 * @brief 更新阶段实现 (Josephs 形式协方差更新)
 * 逻辑：
 * 1. 计算增益 K = P*HT / (H*P*HT + R)
 * 2. 更新协方差 P (采用更为数值稳定的 Joseph form)
 * 3. 更新状态量 x = x + K*residual
 */
Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  Eigen::VectorXd x_prior = x;
  // 1. 计算卡尔曼增益 K
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  Eigen::MatrixXd K = P * H.transpose() * S.inverse();

  // 2. Joseph form 协方差更新：P = (I-KH)P(I-KH)T + KRKT
  // 这种方法比标准的 P = (I-KH)P 具有更好的数值对称性和稳定性。
  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  // 3. 状态更正 (注意使用自定义的 z_subtract 和 x_add)
  x = x_add(x, K * z_subtract(z, h(x)));

  // 4. 发散诊断与卡方检验 (Chi-square test)
  Eigen::VectorXd residual = z_subtract(z, h(x));
  
  // NIS: Normalized Innovation Squared (判定观测是否符合预期分布)
  double nis = residual.transpose() * S.inverse() * residual;
  // NEES: Normalized Estimation Error Squared (判定估计偏差是否在误差椭球内)
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);

  // 检验统计更新
  total_count_++;
  last_nis = nis;
  recent_nis_failures.push_back(nis > 0.711 ? 1 : 0); // 暂定自由度为 4 时的阈值
  if (recent_nis_failures.size() > window_size) recent_nis_failures.pop_front();

  // 数据导出
  data["nis"] = nis;
  data["nees"] = nees;
  data["recent_nis_failures"] = static_cast<double>(std::accumulate(recent_nis_failures.begin(), recent_nis_failures.end(), 0)) / recent_nis_failures.size();

  return x;
}

}  // namespace tools