#ifndef TOOLS__PID_HPP
#define TOOLS__PID_HPP

namespace tools
{

/**
 * @brief 通用 PID 控制器 (Proportional-Integral-Derivative)
 * 逻辑：常规的位置式 PID 模型，支持角度环限域（Angular Mode），用于处理角度突跳问题。
 */
class PID
{
public:
  /**
   * @brief 构造函数
   * @param dt 控制周期 (s)
   * @param kp 比例系数
   * @param ki 积分系数
   * @param kd 微分系数
   * @param max_out 总输出限幅
   * @param max_iout 积分误差限幅 (防止积分饱和)
   * @param angular 是否为角度环模式 (如果是则会自动处理 180/-180 跳变)
   */
  PID(float dt, float kp, float ki, float kd, float max_out, float max_iout, bool angular = false);

  // 调试观测变量
  float pout = 0.0f;  // 比例项输出
  float iout = 0.0f;  // 积分项输出
  float dout = 0.0f;  // 微分项输出

  /**
   * @brief 计算一次 PID 输出信号
   * @param set 期望目标值 (Target)
   * @param fdb 实际反馈值 (Feedback)
   * @return 控制器输出量
   */
  float calc(float set, float fdb);

private:
  const float dt_;
  const float kp_, ki_, kd_;
  const float max_out_, max_iout_;
  const bool angular_;

  float last_fdb_ = 0.0f; // 上一次的反馈值，用于计算微分项
};

}  // namespace tools

#endif  // TOOLS__PID_HPP