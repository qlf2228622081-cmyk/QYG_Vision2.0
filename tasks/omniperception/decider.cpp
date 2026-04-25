#include "decider.cpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace omniperception
{

/**
 * @brief Decider 构造函数
 */
Decider::Decider(const std::string & config_path) : detector_(config_path), count_(0)
{
  auto yaml = YAML::LoadFile(config_path);
  img_width_ = yaml["image_width"].as<double>();
  img_height_ = yaml["image_height"].as<double>();
  fov_h_ = yaml["fov_h"].as<double>();
  fov_v_ = yaml["fov_v"].as<double>();
  new_fov_h_ = yaml["new_fov_h"].as<double>();
  new_fov_v_ = yaml["new_fov_v"].as<double>();
  enemy_color_ =
    (yaml["enemy_color"].as<std::string>() == "red") ? auto_aim::Color::red : auto_aim::Color::blue;
  mode_ = yaml["mode"].as<double>();
}

/**
 * @brief 轮询式决策逻辑
 * 逻辑：每一帧轮流访问 3 个相机（左、右、后），若发现目标则返回运动控制指令。
 */
io::Command Decider::decide(
  auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
  io::USBCamera & usbcam2, io::Camera & back_camera)
{
  Eigen::Vector2d delta_angle;
  io::USBCamera * cams[] = {&usbcam1, &usbcam2};

  cv::Mat usb_img;
  std::chrono::steady_clock::time_point timestamp;
  
  // 轮询计数逻辑
  if (count_ == 2) {
    back_camera.read(usb_img, timestamp);
  } else {
    cams[count_]->read(usb_img, timestamp);
  }
  
  auto armors = yolo.detect(usb_img);
  auto empty = armor_filter(armors);

  if (!empty) {
    // 根据是哪个相机识别到的，计算相对于云台当前朝向的角度差补偿
    if (count_ == 2) {
      delta_angle = this->delta_angle(armors, "back");
    } else {
      delta_angle = this->delta_angle(armors, cams[count_]->device_name);
    }

    tools::logger()->debug(
      "[{} camera] delta yaw:{:.2f},target pitch:{:.2f}",
      (count_ == 2 ? "back" : cams[count_]->device_name), delta_angle[0], delta_angle[1]);

    count_ = (count_ + 1) % 3;

    // 返回绝对角度指令 (当前位置 + 偏差)
    return io::Command{
      true, false, tools::limit_rad(gimbal_pos[0] + delta_angle[0] / 57.3),
      tools::limit_rad(delta_angle[1] / 57.3)};
  }

  count_ = (count_ + 1) % 3;
  return io::Command{false, false, 0, 0};
}

/**
 * @brief 基于全感知并行队列的新决策接口
 */
io::Command Decider::decide(const std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return io::Command{false, false, 0, 0};

  // 取得队列头部（通常已经过外部 sort 排序，即当前优先级最高的目标）
  DetectionResult dr = detection_queue.front();
  if (dr.armors.empty()) return io::Command{false, false, 0, 0};
  
  tools::logger()->info(
    "omniperception find {}, delta yaw is {:.4f}", 
    auto_aim::ARMOR_NAMES[dr.armors.front().name], dr.delta_yaw * 57.3);

  return io::Command{true, false, dr.delta_yaw, dr.delta_pitch};
}

/**
 * @brief 角度补偿逻辑：相机局部检测 -> 云台相对偏航角
 * 逻辑：固定偏移量 (62度 / 170度) 加上 像素位置映射出的角度。
 */
Eigen::Vector2d Decider::delta_angle(
  const std::list<auto_aim::Armor> & armors, const std::string & camera)
{
  Eigen::Vector2d delta_angle;
  if (camera == "left") {
    // 左相机偏移 62 度
    delta_angle[0] = 62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
  }
  else if (camera == "right") {
    // 右相机偏移 -62 度
    delta_angle[0] = -62 + (new_fov_h_ / 2) - armors.front().center_norm.x * new_fov_h_;
    delta_angle[1] = armors.front().center_norm.y * new_fov_v_ - new_fov_v_ / 2;
  }
  else {
    // 后置相机偏移大约 170~180 度
    delta_angle[0] = 170 + (54.2 / 2) - armors.front().center_norm.x * 54.2;
    delta_angle[1] = armors.front().center_norm.y * 44.5 - 44.5 / 2;
  }
  return delta_angle;
}

/**
 * @brief 执行过滤规则
 */
bool Decider::armor_filter(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return true;
  
  // 1. 颜色过滤
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });
  // 2. 编号过滤 (根据当前赛季规则调整)
  armors.remove_if([&](const auto_aim::Armor & a) { 
    return a.name == auto_aim::ArmorName::five || a.name == auto_aim::ArmorName::outpost; 
  });
  // 3. 复活无敌保护过滤
  armors.remove_if([&](const auto_aim::Armor & a) {
    return std::find(invincible_armor_.begin(), invincible_armor_.end(), a.name) !=
           invincible_armor_.end();
  });

  return armors.empty();
}

/**
 * @brief 为扫描到的所有目标赋予静态优先级分数
 */
void Decider::set_priority(std::list<auto_aim::Armor> & armors)
{
  if (armors.empty()) return;
  const PriorityMap & priority_map = (mode_ == MODE_ONE) ? mode1 : mode2;
  for (auto & armor : armors) {
    armor.priority = priority_map.at(armor.name);
  }
}

/**
 * @brief 全场感知数据大排序
 * 逻辑：
 * 1. 滤除无效装甲板。
 * 2. 赋予优先级。
 * 3. 对每个相机的结果内部排序（选出该相机视野内最好的）。
 * 4. 对所有相机的结果进行横向排序（选出全场最好的相机画面）。
 */
void Decider::sort(std::vector<DetectionResult> & detection_queue)
{
  if (detection_queue.empty()) return;

  for (auto & dr : detection_queue) {
    armor_filter(dr.armors);
    set_priority(dr.armors);
    // 相机内部排序
    dr.armors.sort([](const auto_aim::Armor & a, const auto_aim::Armor & b) { 
      return a.priority < b.priority; 
    });
  }

  // 跨相机全局排序
  std::sort(
    detection_queue.begin(), detection_queue.end(),
    [](const DetectionResult & a, const DetectionResult & b) {
      if (a.armors.empty()) return false;
      if (b.armors.empty()) return true;
      return a.armors.front().priority < b.armors.front().priority;
    });
}

/**
 * @brief 接收裁判系统指令：目标无敌状态更新
 */
void Decider::get_invincible_armor(const std::vector<int8_t> & invincible_enemy_ids)
{
  invincible_armor_.clear();
  for (const auto & id : invincible_enemy_ids) {
    tools::logger()->info("invincible armor id: {}", id);
    invincible_armor_.push_back(auto_aim::ArmorName(id - 1)); // 物理 ID 转内部索引
  }
}

}  // namespace omniperception