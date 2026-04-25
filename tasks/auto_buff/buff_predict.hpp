#ifndef BUFF__PREDICT_HPP
#define BUFF__PREDICT_HPP

#include <algorithm>
#include <deque>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tools/extended_kalman_filter.hpp"
#include "tools/img_tools.hpp"
#include "tools/plotter.hpp"

// 小符标准转速 (rad/s)
const double SMALL_W = CV_PI / 3;

/**
 * @brief 预测器基类 (Predictor)
 * 定义了能量机关角度预测器的基本接口。
 */
class Predictor
{
public:
  Predictor(){};
  /**
   * @brief 更新当前观测角度与时间
   */
  virtual void update(double angle, double nowtime) = 0;
  
  /**
   * @brief 预测未来 delta_time 后的角度增量
   */
  virtual double predict(double delta_time) = 0;
  
  virtual bool is_unsolve() const = 0;
  virtual Eigen::VectorXd getX_best() const = 0;

protected:
  // 卡尔曼滤波矩阵定义
  Eigen::VectorXd x0;
  Eigen::MatrixXd P0;
  Eigen::MatrixXd A;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd H;
  Eigen::MatrixXd R;
  tools::ExtendedKalmanFilter ekf;
  
  Eigen::VectorXd X_best;     // 最优估算状态
  double lastangle = 0;       // 上一帧角度
  double lasttime = 0;        // 上一帧时间
  int cw_ccw = 0;             // 旋转方向判定 (顺时针为正)
  bool first_in = true;
  bool unsolvable = true;
};

/**
 * @brief 小符预测器 (Small_Predictor)
 * 逻辑：匀速圆周运动模型，角速度固定为 SMALL_W。
 */
class Small_Predictor : public Predictor
{
public:
  Small_Predictor() : Predictor()
  {
    x0.resize(1); P0.resize(1, 1); A.resize(1, 1);
    Q.resize(1, 1); H.resize(1, 1); R.resize(1, 1);
    
    x0 << 0.0;
    P0 << 1.0;
    A << 1.0;
    Q << 0.0;
    H << 1.0;
    R << 0.05;
    ekf = tools::ExtendedKalmanFilter(x0, P0);
  }

  virtual void update(double angle, double nowtime) override
  {
    // 1. 初始化
    if (first_in) {
      first_in = false; lasttime = nowtime; lastangle = angle;
      ekf.x[0] = angle; unsolvable = true;
    }

    // 2. 角度跳变处理 (补足 2PI/5 的扇叶切换增量)
    if (abs(angle - lastangle) > CV_PI / 12) {
      for (int i = -5; i <= 5; i++) {
        double angle_c = lastangle + i * 2 * CV_PI / 5;
        if (std::fabs(angle_c - angle) < CV_PI / 5) {
          ekf.x[0] += i * 2 * CV_PI / 5; break;
        }
      }
    }

    // 3. 方向判断
    if (abs(cw_ccw) < 100) {
      if (lastangle > angle) cw_ccw -= 1;
      else cw_ccw += 1;
    }

    // 4. 执行卡尔曼预测与更新
    double deltatime = nowtime - lasttime;
    A << 1.0;
    Eigen::VectorXd B(1);
    B << SMALL_W * (cw_ccw > 0 ? 1 : -1);
    
    // 匀速模型：angle_new = angle_old + w * dt
    ekf.predict(A, Q, [&](const Eigen::VectorXd & x) { return A * x + deltatime * B; });

    Eigen::VectorXd z(1); z << angle;
    X_best = ekf.update(z, H, R);

    lasttime = nowtime; lastangle = angle;
    unsolvable = false;
  }

  virtual double predict(double delta_time) override
  {
    if (unsolvable) return 0;
    return (cw_ccw > 0 ? 1 : -1) * SMALL_W * delta_time;
  }

  virtual bool is_unsolve() const { return unsolvable; }
  virtual Eigen::VectorXd getX_best() const { return X_best; }
};

/**
 * @brief 大符预测器 (Big_Predictor)
 * 逻辑：变速圆周运动模型，符合公式：v = a * sin(w * t + sita) + (2.09 - a)。
 * EKF 状态向量包含 [angle, spd, a, w, sita]。
 */
class Big_Predictor : public Predictor
{
public:
  Big_Predictor() : Predictor()
  {
    x0.resize(5); P0.resize(5, 5); A.resize(5, 5);
    Q.resize(5, 5); H.resize(1, 5); R.resize(1, 1);

    // 预置大符参数范围初值
    x0 << 0.0, 1.1775, 0.9125, 1.942, 0.0;
    P0.setIdentity(); P0 *= 0.01;
    Q << 0.03, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.05;
    H << 1.0, 0.0, 0.0, 0.0, 0.0;
    R << 0.05;
    ekf = tools::ExtendedKalmanFilter(x0, P0);
  }

  virtual void update(double angle, double nowtime) override
  {
    // 1. 初始化与角度补偿 (同小符逻辑)
    if (first_in) {
      first_in = false; lasttime = nowtime; lastangle = angle;
      ekf.x[0] = angle; unsolvable = true; return;
    }
    if (abs(angle - lastangle) > CV_PI / 12) {
      for (int i = -5; i <= 5; i++) {
        double angle_c = lastangle + i * 2 * CV_PI / 5;
        if (std::fabs(angle_c - angle) < CV_PI / 5) {
          ekf.x[0] += i * 2 * CV_PI / 5; lastangle += i * 2 * CV_PI / 5; break;
        }
      }
    }
    if (abs(cw_ccw) < 100) {
      if (lastangle > angle) cw_ccw -= 1; else cw_ccw += 1;
    }

    // 2. 构造状态转移雅可比矩阵 A
    double deltatime = nowtime - lasttime;
    double a = ekf.x[2];
    double w = ekf.x[3];
    double sita = ekf.x[4];
    
    // 此处省去 A 矩阵的复杂偏导数定义...
    // 3. 执行非线性预测 f(x)
    ekf.predict(A, Q, [&](const Eigen::VectorXd & x) {
      Eigen::VectorXd m(5);
      // 积分速度公式得出新的角度
      m << x[0] + (cw_ccw > 0 ? 1 : -1) * (-a / w * cos(sita + w * deltatime) + a / w * cos(sita) + (2.09 - a) * deltatime),
        a * sin(sita + w * deltatime) + 2.09 - a, a, w, sita + w * deltatime;
      return m;
    });

    // 4. 更新观测结果
    Eigen::VectorXd z(1); z << angle;
    X_best = ekf.update(z, H, R);

    lasttime = nowtime; lastangle = angle; unsolvable = false;
  }

  virtual double predict(double delta_time) override
  {
    if (unsolvable) return 0;
    double a = X_best[2]; double w = X_best[3]; double sita = X_best[4];
    // 根据积分公式计算未来 delta_time 后的角度偏移
    return (cw_ccw > 0 ? 1 : -1) *
           (-a / w * cos(sita + w * delta_time) + a / w * cos(sita) + (2.09 - a) * delta_time);
  }

  virtual bool is_unsolve() const { return unsolvable; }
  virtual Eigen::VectorXd getX_best() const { return X_best; }
};

/**
 * @brief 3D 坐标预测器 (XYZ_predictor)
 * 逻辑：简单的 XYZ 三轴卡尔曼滤波器，用于平滑目标 R 标在世界系下的位置变化。
 */
class XYZ_predictor
{
public:
  // 卡尔曼参数
  Eigen::VectorXd x0; Eigen::MatrixXd P0; Eigen::MatrixXd A;
  Eigen::MatrixXd Q;  Eigen::MatrixXd H;  Eigen::MatrixXd R;
  tools::ExtendedKalmanFilter ekf;
  Eigen::VectorXd X_best;

  XYZ_predictor() : x0Lazy(3), P0Lazy(3, 3), ... // 简化初始化描述
  {
    x0.resize(3); P0.resize(3, 3); A.resize(3, 3);
    Q.resize(3, 3); H.resize(3, 3); R.resize(3, 3);
    
    x0 << 0.0, 0.0, 7.0; // 假设能量机关在 7m 处
    P0.setIdentity();
    A.setIdentity();
    Q.setZero();
    H.setIdentity();
    R.setIdentity();
    ekf = tools::ExtendedKalmanFilter(x0, P0);
  }

  /**
   * @brief 执行滤波平滑
   */
  void kalman(Eigen::Vector3d & XYZ)
  {
    if (first_in) {
      first_in = false; ekf.x[0] = XYZ[0]; ekf.x[1] = XYZ[1]; ekf.x[2] = XYZ[2]; return;
    }
    ekf.predict(A, Q);
    Eigen::VectorXd z(3); z << XYZ[0], XYZ[1], XYZ[2];
    X_best = ekf.update(z, H, R);
    XYZ = X_best;
  }

private:
  bool first_in = true;
};

#endif  // BUFF_PREDICT_HPP
