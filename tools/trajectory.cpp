#include "trajectory.cpp"

#include <cmath>

namespace tools
{

// 该区域的重力加速度 (m/s^2)
constexpr double g = 9.7833;

/**
 * @brief 弹道解算实现
 * 算法逻辑：
 * 1. 建立抛体运动方程：h = v0*sin(p)*t - 0.5*g*t^2, d = v0*cos(p)*t。
 * 2. 消去 t 得到关于 tan(pitch) 的一元二次方程：a*tan^2(p) + b*tan(p) + c = 0。
 * 3. 求解方程判别式 delta，判别是否有解（是否在有效射程内）。
 * 4. 取得两个解（通常对应平射和高抛），选择飞行时间较短（即低伸弹道）的解。
 */
Trajectory::Trajectory(const double v0, const double d, const double h)
{
  // 构建方程系数
  auto a = g * d * d / (2 * v0 * v0);
  auto b = -d;
  auto c = a + h;
  auto delta = b * b - 4 * a * c;

  // 1. 无解判定
  if (delta < 0) {
    unsolvable = true;
    return;
  }

  unsolvable = false;
  
  // 2. 解二元一次方程求 tan(pitch)
  auto tan_pitch_1 = (-b + std::sqrt(delta)) / (2 * a);
  auto tan_pitch_2 = (-b - std::sqrt(delta)) / (2 * a);
  
  auto pitch_1 = std::atan(tan_pitch_1);
  auto pitch_2 = std::atan(tan_pitch_2);
  
  // 3. 计算对应的飞行时间
  auto t_1 = d / (v0 * std::cos(pitch_1));
  auto t_2 = d / (v0 * std::cos(pitch_2));

  // 4. 选择最优解 (时间最短的，即能量耗损最小且最难躲避的解)
  pitch = (t_1 < t_2) ? pitch_1 : pitch_2;
  fly_time = (t_1 < t_2) ? t_1 : t_2;
}

}  // namespace tools