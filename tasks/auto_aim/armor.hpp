#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{
enum Color
{
  red,
  blue,
  extinguish,
  purple
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};

enum ArmorType
{
  big,
  small
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};

enum ArmorName
{
  one,
  two,
  three,
  four,
  five,
  sentry,
  outpost,
  base,
  not_armor
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};

enum ArmorPriority
{
  first = 1,
  second,
  third,
  forth,
  fifth
};

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

struct Lightbar
{
  std::size_t id;
  Color color;
  cv::Point2f center, top, bottom, top2bottom;
  std::vector<cv::Point2f> points;
  double angle, angle_error, length, width, ratio;
  cv::RotatedRect rotated_rect;

  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  Lightbar() {};
};

struct Armor
{
  /*
  Armor 是自瞄链路里的统一“单帧装甲板观测对象”。
  不同检测前端（传统灯条法 / YOLO）都会先构造成 Armor，
  然后再由后处理、Solver、Tracker 逐步补全字段。
  所以不是所有成员一创建就都有意义，要看当前处理流水线走到了哪一步。
  */

  // ===== 基本标签与二维几何观测 =====
  Color color;  // 颜色标签。Tracker 会据此过滤敌我装甲板，调试窗口也会直接显示它。
  Lightbar left, right;  // 传统灯条法下组成该装甲板的左右灯条。Detector 会用它们裁图案 ROI、
                         // 检查共用灯条、估计大小装甲；纯 YOLO 路径下这两个字段通常不参与后续计算。
  cv::Point2f center;  // 图像平面中的几何中心，通常是四角点平均值。Tracker 用它按离图像中心远近排序，
                       // 调试绘图也常拿它当文字锚点。它不是严格物理中心，只是二维观测中心。
  cv::Point2f center_norm;  // 把 center 归一化到 [0, 1] 图像坐标系。这样不同分辨率下可统一比较位置，
                            // omniperception 等模块会直接用它算视场偏角。
  std::vector<cv::Point2f> points;  // 四个角点（通常按左上、右上、右下、左下顺序）。
                                    // 这是最关键的二维输入，Solver::solve 会用它做 solvePnP，
                                    // 传统二次矫正也会直接修改它。

  // ===== 传统几何筛选特征 =====
  double ratio;  // 装甲板宽高/跨度比例特征。
                 // 传统 Detector 会用它筛掉明显不合法的装甲板，也会辅助判断 big / small。
  double side_ratio;  // 左右两边“高度”是否一致。两侧灯条长度差太大时，通常说明配对不可信。
  double rectangular_error;  // 装甲板偏离标准矩形的程度。
                             // Detector 会用它判断这是不是一块像样的矩形装甲板。

  // ===== 分类与检测输出信息 =====
  ArmorType type;  // big / small。Solver 需要它来选择对应的三维装甲板模型点，
                   // Tracker 也会用它和 name 一起做目标匹配。
  ArmorName name;  // 编号或类别：1/2/3/4/5、sentry、outpost、base 等。
                   // Tracker 会根据 name 决定初始化哪种 Target，Aimer/Target 也会据此选不同策略。
  ArmorPriority priority;  // 目标优先级，数值越小优先级越高。
                           // Tracker 会按它排序、切目标，所以它属于“决策信息”而不是纯感知信息。
  int class_id;  // 神经网络原始类别 id。主要用于“网络输出 -> color/name/type”这一步映射，
                 // 也方便后续调试模型类别定义是否对齐。
  cv::Rect box;  // 二维检测框。主要保存网络输出的 bbox，调试显示、ROI 推理和日志里会用到。
  cv::Mat pattern;  // 从原图裁出来的图案 ROI。Classifier 依赖它做数字分类/复分类，
                    // 调试时也常把它单独存图观察识别效果。
  double confidence;  // 该装甲板检测/分类的置信度。Detector/YOLO 会据此过滤低质量结果，
                      // 调试输出也会直接显示这个分数。
  bool duplicated;  // 是否与其它装甲板重复（例如共用灯条、重叠候选）。
                    // Detector 后处理会先把重复候选标记为 true，再统一删除。

  // ===== 单帧三维解算结果（由 Solver::solve 补全） =====
  Eigen::Vector3d xyz_in_gimbal;  // 装甲板中心在云台坐标系下的位置，单位 m。
                                  // 后续若要和云台控制量、炮口方向直接关联，通常先看这个量。
  Eigen::Vector3d xyz_in_world;   // 装甲板中心在世界坐标系下的位置，单位 m。
                                  // Tracker/Target 会基于它建立目标运动模型。
  Eigen::Vector3d ypr_in_gimbal;  // 装甲板在云台坐标系下的 yaw/pitch/roll，单位 rad。
                                  // 更适合看“相对云台”的当前姿态。
  Eigen::Vector3d ypr_in_world;   // 装甲板在世界坐标系下的 yaw/pitch/roll，单位 rad。
                                  // Target::update 会直接用这个 yaw 参与旋转目标匹配。
  Eigen::Vector3d ypd_in_world;   // 装甲板在世界坐标系下的球坐标表示（yaw, pitch, distance）。
                                  // EKF 更新时会把它当作更稳定的观测量之一。

  double yaw_raw;  // yaw 优化前的原始值，单位 rad。
                   // Solver 会先求出初值，再可能做重投影优化；这个字段用于保留优化前结果做调试对比。

  //以下是Armor的构造函数声明，而且是5个重载构造函数
  //因为名字和类型名一样，都是Armor，所以构造函数不能写成Armor()，而是要写成Armor(参数列表)，参数列表不同就构成了重载
  //没有返回值类型，构造函数的作用是根据输入参数初始化一个Armor对象
  //写在struct内部，说明它们是这个结构体的成员函数，可以直接访问结构体的成员变量
  
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
