#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>
#include <iostream>
#include <list>
#include <unordered_map>

#include "detection.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace omniperception
{

/**
 * @brief 行为决策器 (Decider)
 * 逻辑：负责根据全场多个相机的感知结果，进行目标过滤、优先级排序，并做出最终
 * 的打击决策（选择攻击哪一个装甲板）。
 * 支持不同的优先级模式（如：英雄优先、哨兵优先等）。
 */
class Decider
{
public:
  /**
   * @brief 构造函数
   */
  Decider(const std::string & config_path);

  /**
   * @brief 多相机整合决策逻辑 (旧模式：轮询)
   */
  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
    io::USBCamera & usbcam2, io::Camera & back_cammera);

  /**
   * @brief 单后置相机决策逻辑
   */
  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera);

  /**
   * @brief 异步感知队列决策逻辑 (新模式)
   * 从多个感知结果中挑选出优先级最高的目标。
   */
  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  /**
   * @brief 计算相机坐标系下目标点相对于中心的偏移角度
   */
  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  /**
   * @brief 装甲板过滤器：滤除友军、无效编号（如5号、前哨站等）及无敌状态目标
   * @return 如果过滤后列表为空，返回 true
   */
  bool armor_filter(std::list<auto_aim::Armor> & armors);

  /**
   * @brief 设置装甲板打击优先级
   */
  void set_priority(std::list<auto_aim::Armor> & armors);

  /**
   * @brief 对感知队列进行全局过滤与排序
   */
  void sort(std::vector<DetectionResult> & detection_queue);

  /**
   * @brief 获取当前锁定目标的详细信息（供通信层下发）
   */
  Eigen::Vector4d get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

  /**
   * @brief 更新无敌状态列表 (通常由裁判系统信号驱动)
   */
  void get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids);

  /**
   * @brief 更新来自导航层的目标锁定要求
   */
  void get_auto_aim_target(
    std::list<auto_aim::Armor> & armors, const std::vector<int8_t> & auto_aim_target);

private:
  int img_width_;
  int img_height_;
  double fov_h_, new_fov_h_; // 相机水平视场角
  double fov_v_, new_fov_v_; // 相机垂直视场角
  int mode_;                 // 打击优先级模式
  int count_;                // 感知设备轮询计数

  auto_aim::Color enemy_color_;
  auto_aim::YOLO detector_;
  std::vector<auto_aim::ArmorName> invincible_armor_;  // 处于无敌状态的装甲板 ID

  // 定义目标优先级映射表
  using PriorityMap = std::unordered_map<auto_aim::ArmorName, auto_aim::ArmorPriority>;

  // 模式 1 优先级：3/4号(步兵) > 1号(英雄) > 2号(工程)/哨兵 > 基地/前哨站
  const PriorityMap mode1 = {
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::forth},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::fifth},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::fifth}};

  // 模式 2 优先级：2号(工程) > 1/3/4/5号 > 哨兵/前哨站/基地
  const PriorityMap mode2 = {
    {auto_aim::ArmorName::two, auto_aim::ArmorPriority::first},
    {auto_aim::ArmorName::one, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::three, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::four, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::five, auto_aim::ArmorPriority::second},
    {auto_aim::ArmorName::sentry, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::outpost, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::base, auto_aim::ArmorPriority::third},
    {auto_aim::ArmorName::not_armor, auto_aim::ArmorPriority::third}};
};

/**
 * @brief 优先级决策模式枚举
 */
enum PriorityMode
{
  MODE_ONE = 1,
  MODE_TWO
};

}  // namespace omniperception

#endif