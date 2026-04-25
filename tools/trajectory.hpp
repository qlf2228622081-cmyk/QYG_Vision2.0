#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{

/**
 * @brief 弹道解算器 (Trajectory)
 * 逻辑：基于物理模型（当前实现为忽略空气阻力的抛体模型），根据目标距离与垂直高度差，
 * 反算应当抬升的角度（Pitch）以及子弹在空中的飞行时间。
 */
struct Trajectory
{
  bool unsolvable; // 是否无解（超出射程）
  double fly_time; // 子弹飞行时间 (s)
  double pitch;    // 应瞄准的俯仰角 (单位：rad, 抬头为正)

  /**
   * @brief 弹道构造函数（执行计算）
   * @param v0 子弹初速度 (m/s)
   * @param d 目标相对于枪口的水平距离 (m)
   * @param h 目标相对于枪口的垂直坐标 (m)
   */
  Trajectory(const double v0, const double d, const double h);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP