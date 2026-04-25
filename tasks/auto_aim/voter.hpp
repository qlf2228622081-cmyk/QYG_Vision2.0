#ifndef AUTO_AIM__VOTER_HPP
#define AUTO_AIM__VOTER_HPP

#include <vector>

#include "armor.hpp"

namespace auto_aim
{

/**
 * @brief 投票器类 (Voter)
 * 逻辑：用于对目标的属性（颜色、兵种、大小）进行统计投票。
 * 通过多帧累计结果，减少单帧识别误差带来的抖动。
 */
class Voter
{
public:
  Voter();

  /**
   * @brief 为特定的属性组合投一票
   */
  void vote(const Color color, const ArmorName name, const ArmorType type);

  /**
   * @brief 获取特定属性组合的当前票数
   */
  std::size_t count(const Color color, const ArmorName name, const ArmorType type);

private:
  std::vector<std::size_t> count_; // 扁平化的投票计数数组

  /**
   * @brief 内部索引映射函数
   * 将 (Color, Name, Type) 三元组映射到一维数组索引。
   */
  std::size_t index(const Color color, const ArmorName name, const ArmorType type) const;
};
}  // namespace auto_aim

#endif