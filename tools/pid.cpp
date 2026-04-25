#include "pid.cpp"

#include "math_tools.hpp"

// 辅助限幅函数
float clip(float value, float min, float max) { return std::max(min, std::min(max, value)); }

namespace tools
{

/**
 * @brief PID 构造函数
 */
PID::PID(float dt, float kp, float ki, float kd, float max_out, float max_iout, bool angular)
: dt_(dt), kp_(kp), ki_(ki), kd_(kd), max_out_(max_out), max_iout_(max_iout), angular_(angular)
{
}

/**
 * @brief PID 计算逻辑实现
 * 1. 处理偏差 e = set - fdb (角度模式下进行弧度限域)。
 * 2. 处理反馈微分项 de = last_fdb - fdb (减少控制环对设定值阶跃的抖动敏感度)。
 * 3. 累积积分项并执行反饱和限制 (Anti-windup)。
 * 4. 返回加权总输出。
 */
float PID::calc(float set, float fdb)
{
  // 1. 计算偏差 (Error)
  float e = angular_ ? limit_rad(set - fdb) : (set - fdb);
  // 2. 计算反馈导数 (Feedback Derivative) 为了避免设定点跳变引起微分冲击
  float de = angular_ ? limit_rad(last_fdb_ - fdb) : (last_fdb_ - fdb);
  last_fdb_ = fdb;

  // 比例项：P = Kp * e
  this->pout = e * kp_;
  
  // 积分项：I = Integral(e * dt) * Ki
  // 包含积分限幅逻辑，防止积分风暴
  this->iout = clip(this->iout + e * dt_ * ki_, -max_iout_, max_iout_);
  
  // 微分项：D = (de / dt) * Kd
  this->dout = de / dt_ * kd_;

  // 加总并执行全局输出限幅
  return clip(this->pout + this->iout + this->dout, -max_out_, max_out_);
}

}  // namespace tools
