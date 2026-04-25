#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{

/**
 * @brief 目标颜色枚举
 */
enum Color
{
  red,        // 红色
  blue,       // 蓝色
  extinguish, // 灭灯 (通常指已击毁的目标或非比赛颜色)
  purple      // 紫色 (某些特殊规则下的颜色)
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

/**
 * @brief 装甲板尺寸类型枚举
 */
enum ArmorType
{
  big,   // 大装甲板 (230mm)
  small  // 小装甲板 (135mm)
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

/**
 * @brief 目标兵种/编号枚举
 */
enum ArmorName
{
  one,      // 1号: 英雄
  two,      // 2号: 工程
  three,    // 3号: 步兵
  four,     // 4号: 步兵
  five,     // 5号: 步兵
  sentry,   // 哨兵
  outpost,  // 前哨站
  base,     // 基地
  not_armor // 非装甲板 (干扰项)
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                               "sentry", "outpost", "base",  "not_armor"};

/**
 * @brief 攻击优先级枚举 (数字越小优先级越高)
 */
enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

/**
 * @brief 预置的装甲板属性表 (颜色, 名称, 类型)
 * 用于神经网络输出的 class_id 到映射具体兵种。
 */
// clang-format off
const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
  {blue, sentry, small},     {red, sentry, small},     {extinguish, sentry, small},
  {blue, one, small},        {red, one, small},        {extinguish, one, small},
  {blue, two, small},        {red, two, small},        {extinguish, two, small},
  {blue, three, small},      {red, three, small},      {extinguish, three, small},
  {blue, four, small},       {red, four, small},       {extinguish, four, small},
  {blue, five, small},       {red, five, small},       {extinguish, five, small},
  {blue, outpost, small},    {red, outpost, small},    {extinguish, outpost, small},
  {blue, base, big},         {red, base, big},         {extinguish, base, big},      {purple, base, big},       
  {blue, base, small},       {red, base, small},       {extinguish, base, small},    {purple, base, small},    
  {blue, three, big},        {red, three, big},        {extinguish, three, big}, 
  {blue, four, big},         {red, four, big},         {extinguish, four, big},  
  {blue, five, big},         {red, five, big},         {extinguish, five, big}};
// clang-format on

/**
 * @brief 灯条结构体 (Lightbar)
 * 封装传统几何视觉提取的灯条特征。
 */
struct Lightbar
{
  std::size_t id;          // 调试 ID
  Color color;             // 灯条颜色
  cv::Point2f center, top, bottom, top2bottom; // 特征点：中心、上端、下端、方向向量
  std::vector<cv::Point2f> points; // 包含上下端点的列表
  double angle, angle_error, length, width, ratio; // 几何属性：倾角、误差、长度、宽度、长宽比
  cv::RotatedRect rotated_rect;   // OpenCV 旋转外接矩形

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar() {};
};

/**
 * @brief 装甲板结构体 (Armor)
 * 封装了装甲板的所有动态特征（几何信息、姿态信息、分类信息）。
 */
struct Armor
{
  Color color;             // 装甲板最终判定的颜色
  Lightbar left, right;    // 左右灯条 (仅传统视觉有效)
  cv::Point2f center;      // 图像坐标系下的几何中心
  cv::Point2f center_norm; // 归一化后的中心坐标 ([-1, 1])
  std::vector<cv::Point2f> points; // 4 个角点的像素坐标 (TL, TR, BR, BL)

  // 几何筛选特征
  double ratio;              // 宽度与高度之比
  double side_ratio;         // 两灯条长度比例
  double rectangular_error;  // 矩形程度误差

  ArmorType type;            // 装甲板类型 (大/小)
  ArmorName name;            // 兵种名称 (数字/基地等)
  ArmorPriority priority;    // 攻击优先级
  int class_id;              // 神经网络对应的类别 ID
  cv::Rect box;              // 装甲板外接矩形 (ROI)
  cv::Mat pattern;           // 数字区域图像 (用于分类)
  double confidence;         // 检测置信度
  bool duplicated;           // 是否为重复检测单元

  // 空间位姿信息 (解算结果)
  Eigen::Vector3d xyz_in_gimbal;  // 相对于云台中心的 3D 坐标 (x-前, y-左, z-上), 单位：m
  Eigen::Vector3d xyz_in_world;   // 世界坐标系 (IMU) 下的 3D 坐标, 单位：m
  Eigen::Vector3d ypr_in_gimbal;  // 云台系下的角度 (Yaw, Pitch, Roll), 单位：rad
  Eigen::Vector3d ypr_in_world;   // 世界系下的角度, 单位：rad
  Eigen::Vector3d ypd_in_world;   // 世界系下的球坐标 (Yaw, Pitch, Distance)

  double yaw_raw;  // 优化前的原始 Yaw 角

  // 构造函数
  Armor(const Lightbar & left, const Lightbar & right);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
    cv::Point2f offset);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints);
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP