#include "voter.hpp"

namespace auto_aim
{

/**
 * @brief Voter 构造函数
 * 初始化计数数组，大小等于所有可能属性组合的总数。
 */
Voter::Voter() : count_(COLORS.size() * ARMOR_NAMES.size() * ARMOR_TYPES.size(), 0) {}

/**
 * @brief 在对应位置累加票数
 */
void Voter::vote(const Color color, const ArmorName name, const ArmorType type)
{
  count_[index(color, name, type)] += 1;
}

/**
 * @brief 查询指定组合的得票数
 */
std::size_t Voter::count(const Color color, const ArmorName name, const ArmorType type)
{
  return count_[index(color, name, type)];
}

/**
 * @brief 核心索引转换函数
 * 将 3D 属性空间映射到 1D 数组索引：
 * index = color * (NAME_SIZE * TYPE_SIZE) + name * TYPE_SIZE + type
 */
std::size_t Voter::index(const Color color, const ArmorName name, const ArmorType type) const
{
  return color * (ARMOR_NAMES.size() * ARMOR_TYPES.size()) + name * ARMOR_TYPES.size() + type;
}

} // namespace auto_aim