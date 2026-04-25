#include "detector.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

/**
 * @brief Detector 构造函数
 * 从参数文件中加载各类几何约束阈值，如灯条长宽比、装甲板比例等。
 */
Detector::Detector(const std::string & config_path, bool debug)
: classifier_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  // 加载各类阈值参数
  threshold_ = yaml["threshold"].as<double>();
  max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3;  // 度转弧度
  min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
  max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
  min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
  min_armor_ratio_ = yaml["min_armor_ratio"].as<double>();
  max_armor_ratio_ = yaml["max_armor_ratio"].as<double>();
  max_side_ratio_ = yaml["max_side_ratio"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  max_rectangular_error_ = yaml["max_rectangular_error"].as<double>() / 57.3;

  // 样本保存路径
  save_path_ = "patterns";
  std::filesystem::create_directory(save_path_);
}

/**
 * @brief 传统自瞄主检测函数 (全图扫描)
 * 流程：灰度化 -> 二值化 -> 轮廓提取 -> 灯条过滤 -> 装甲板配对 -> 数字分类 -> 重叠拆除
 */
std::list<Armor> Detector::detect(const cv::Mat & bgr_img, int frame_count)
{
  // 1. 预处理：彩色图转灰度图并进行二值化
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
  
  if (debug_) cv::imshow("binary_img", binary_img); // 调试显示二值化结果

  // 2. 提取轮廓点并转化为初始灯条对象
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);      // 最小外接矩形
    auto lightbar = Lightbar(rotated_rect, lightbar_id);

    // 几何特征检查（长度、比例、倾角）
    if (!check_geometry(lightbar)) continue;

    lightbar.color = get_color(bgr_img, contour); // 判定颜色（红/蓝）
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  // 3. 灯条预排序
  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // 4. 灯条两两配对，拟合装甲板候选
  std::list<Armor> armors;
  for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
    for (auto right = std::next(left); right != lightbars.end(); right++) {
      if (left->color != right->color) continue; // 颜色须一致

      auto armor = Armor(*left, *right);
      
      // 检查装甲板几何比例是否合法
      if (!check_geometry(armor)) continue;

      // 提取中间数字图案并进行 CNN 分类
      armor.pattern = get_pattern(bgr_img, armor);
      classifier_.classify(armor);
      
      // 过滤非装甲板（0号/空装甲）及低置信度目标
      if (!check_name(armor)) continue;

      // 根据物理比例判定大小装甲板类别
      armor.type = get_type(armor);
      if (!check_type(armor)) continue;

      // 最终确认：转换归一化中心坐标
      armor.center_norm = get_center_norm(bgr_img, armor.center);
      armors.emplace_back(armor);
    }
  }

  // 5. 冲突检查与拆除（处理多个装甲板共用一个灯条的情况）
  for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
    for (auto armor2 = std::next(armor1); armor2 != armors.end(); armor2++) {
      // 若两个装甲板完全不共用任何灯条，跳过
      if (
        armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
        armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
        continue;
      }

      // 如果是共用一个边缘灯条（装甲板重叠），保留投影图案更大的一个（通常更近或更正）
      if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
        auto area1 = armor1->pattern.cols * armor1->pattern.rows;
        auto area2 = armor2->pattern.cols * armor2->pattern.rows;
        if (area1 < area2)
          armor2->duplicated = true;
        else
          armor1->duplicated = true;
      }

      // 如果灯条相连，优先保留分类置信度最高的一个
      if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
        if (armor1->confidence < armor2->confidence)
          armor1->duplicated = true;
        else
          armor2->duplicated = true;
      }
    }
  }

  // 移除标记为冗余的目标
  armors.remove_if([&](const Armor & a) { return a.duplicated; });

  // 调试辅助显示
  if (debug_) show_result(binary_img, bgr_img, lightbars, armors, frame_count);

  return armors;
}

/**
 * @brief 局部探测/微调逻辑
 * 针对已知装甲板的 ROI 区域进行精细化识别，提高角点精度。
 */
bool Detector::detect(Armor & armor, const cv::Mat & bgr_img)
{
  // 1. 构建候选区域关键点并计算 BoundingBox
  auto tl = armor.points[0];
  auto tr = armor.points[1];
  auto br = armor.points[2];
  auto bl = armor.points[3];
  
  // 几何推算：稍微扩大并调整识别区域
  auto lt2b = bl - tl;
  auto rt2b = br - tr;
  auto tl1 = (tl + bl) / 2 - lt2b;
  auto bl1 = (tl + bl) / 2 + lt2b;
  auto br1 = (tr + br) / 2 + rt2b;
  auto tr1 = (tr + br) / 2 - rt2b;
  auto tl2tr = tr1 - tl1;
  auto bl2br = br1 - bl1;
  auto tl2 = (tl1 + tr) / 2 - 0.75 * tl2tr;
  auto tr2 = (tl1 + tr) / 2 + 0.75 * tl2tr;
  auto bl2 = (bl1 + br) / 2 - 0.75 * bl2br;
  auto br2 = (bl1 + br) / 2 + 0.75 * bl2br;

  std::vector<cv::Point> points = {tl2, tr2, br2, bl2};
  auto armor_rotaterect = cv::minAreaRect(points);
  cv::Rect boundingBox = armor_rotaterect.boundingRect();
  
  // 越界检查
  if (
    boundingBox.x < 0 || boundingBox.y < 0 || boundingBox.x + boundingBox.width > bgr_img.cols ||
    boundingBox.y + boundingBox.height > bgr_img.rows) {
    return false;
  }

  // 2. 局部图像预处理
  cv::Mat armor_roi = bgr_img(boundingBox);
  if (armor_roi.empty()) return false;

  cv::Mat gray_img, binary_img;
  cv::cvtColor(armor_roi, gray_img, cv::COLOR_BGR2GRAY);
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);

  // 3. 提取局部灯条并寻找最近邻匹配点
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
  
  std::size_t lightbar_id = 0;
  std::list<Lightbar> lightbars;
  for (const auto & contour : contours) {
    auto rotated_rect = cv::minAreaRect(contour);
    auto lightbar = Lightbar(rotated_rect, lightbar_id);
    if (!check_geometry(lightbar)) continue;
    lightbar.color = get_color(bgr_img, contour);
    lightbars.emplace_back(lightbar);
    lightbar_id += 1;
  }

  if (lightbars.size() < 2) return false;

  lightbars.sort([](const Lightbar & a, const Lightbar & b) { return a.center.x < b.center.x; });

  // 4. 最近邻关联：找到与上一帧最接近的左右灯条
  Lightbar * closest_left_lightbar = nullptr;
  Lightbar * closest_right_lightbar = nullptr;
  float min_distance_tl_bl = std::numeric_limits<float>::max();
  float min_distance_br_tr = std::numeric_limits<float>::max();
  
  for (auto & lightbar : lightbars) {
    // 计算灯条四个端点与 ROI 内检测结果的距离和
    float distance_tl_bl =
      cv::norm(tl - (lightbar.top + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(bl - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y)));
    if (distance_tl_bl < min_distance_tl_bl) {
      min_distance_tl_bl = distance_tl_bl;
      closest_left_lightbar = &lightbar;
    }
    float distance_br_tr =
      cv::norm(br - (lightbar.bottom + cv::Point2f(boundingBox.x, boundingBox.y))) +
      cv::norm(tr - (lightbar.top + cv::Point2f(boundingBox.x, Huby(boundingBox.y))));
    if (distance_br_tr < min_distance_br_tr) {
      min_distance_br_tr = distance_br_tr;
      closest_right_lightbar = &lightbar;
    }
  }

  // 若关联成功且距离误差在可控范围内，则更新角点
  if (
    closest_left_lightbar && closest_right_lightbar &&
    min_distance_br_tr + min_distance_tl_bl < 15) {
    armor.points[0] = closest_left_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[1] = closest_right_lightbar->top + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[2] = closest_right_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    armor.points[3] = closest_left_lightbar->bottom + cv::Point2f(boundingBox.x, boundingBox.y);
    return true;
  }

  return false;
}

/**
 * @brief 灯条几何约束检查
 */
bool Detector::check_geometry(const Lightbar & lightbar) const
{
  auto angle_ok = lightbar.angle_error < max_angle_error_;
  auto ratio_ok = lightbar.ratio > min_lightbar_ratio_ && lightbar.ratio < max_lightbar_ratio_;
  auto length_ok = lightbar.length > min_lightbar_length_;
  return angle_ok && ratio_ok && length_ok;
}

/**
 * @brief 装甲板几何约束检查
 */
bool Detector::check_geometry(const Armor & armor) const
{
  auto ratio_ok = armor.ratio > min_armor_ratio_ && armor.ratio < max_armor_ratio_;
  auto side_ratio_ok = armor.side_ratio < max_side_ratio_;
  auto rectangular_error_ok = armor.rectangular_error < max_rectangular_error_;
  return ratio_ok && side_ratio_ok && rectangular_error_ok;
}

/**
 * @brief 识别名称与置信度检查
 */
bool Detector::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于后续分类器离线迭代
  if (name_ok && !confidence_ok) save(armor);

  if (armor.name == ArmorName::five) tools::logger()->debug("See pattern 5");

  return name_ok && confidence_ok;
}

/**
 * @brief 装甲板大小逻辑一致性检查 (防止错认)
 */
bool Detector::check_type(const Armor & armor) const
{
  auto name_ok = armor.type == ArmorType::small
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name == ArmorName::one || armor.name == ArmorName::base);

  if (!name_ok) {
    tools::logger()->debug(
      "see strange armor: {} {}", ARMOR_TYPES[armor.type], ARMOR_NAMES[armor.name]);
    save(armor);
  }

  return name_ok;
}

/**
 * @brief 判定颜色逻辑：计算轮廓内红色与蓝色通道的累加均值
 */
Color Detector::get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const
{
  int red_sum = 0, blue_sum = 0;

  for (const auto & point : contour) {
    red_sum += bgr_img.at<cv::Vec3b>(point)[2];
    blue_sum += bgr_img.at<cv::Vec3b>(point)[0];
  }

  return blue_sum > red_sum ? Color::blue : Color::red;
}

/**
 * @brief 提取装甲板中心图案 (Perspective Crop) 用于数字分类器输入
 */
cv::Mat Detector::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  // 1. 根据灯条位置推算理想的正交矩形角点
  // 这里的 1.125 系数是基于装甲板物理尺寸与灯条长度的固定比例 (126mm / 56mm / 2)
  auto tl = armor.left.center - armor.left.top2bottom * 1.125;
  auto bl = armor.left.center + armor.left.top2bottom * 1.125;
  auto tr = armor.right.center - armor.right.top2bottom * 1.125;
  auto br = armor.right.center + armor.right.top2bottom * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  return bgr_img(roi);
}

/**
 * @brief 判定大小装甲板
 */
ArmorType Detector::get_type(const Armor & armor)
{
  // 根据长宽比初步判定
  if (armor.ratio > 3.0) return ArmorType::big;
  if (armor.ratio < 2.5) return ArmorType::small;

  // 根据预设的规则强制绑定（如 1号 必定是大装甲）
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
    return ArmorType::big;
  }
  return ArmorType::small;
}

/**
 * @brief 坐标归一化
 */
cv::Point2f Detector::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

/**
 * @brief 保存样本图片到磁盘
 */
void Detector::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, armor.pattern);
}

/**
 * @brief 调试绘制函数
 */
void Detector::show_result(
  const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
  const std::list<Armor> & armors, int frame_count) const
{
  auto detection = bgr_img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

  // 绘制灯条信息
  for (const auto & lightbar : lightbars) {
    auto info = fmt::format(
      "{:.1f} {:.1f} {:.1f} {}", lightbar.angle_error * 57.3, lightbar.ratio, lightbar.length,
      COLORS[lightbar.color]);
    tools::draw_text(detection, info, lightbar.top, {0, 255, 255});
    tools::draw_points(detection, lightbar.points, {0, 255, 255}, 3);
  }

  // 绘制识别出的装甲板信息
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {:.2f} {:.1f} {:.2f} {} {}", armor.ratio, armor.side_ratio,
      armor.rectangular_error * 57.3, armor.confidence, ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.left.bottom, {0, 255, 0});
  }

  cv::Mat binary_img2, detection2;
  cv::resize(binary_img, binary_img2, {}, 0.5, 0.5);
  cv::resize(detection, detection2, {}, 0.5, 0.5);

  cv::imshow("detection", detection2);
}

/**
 * @brief 灯条角点校正器 (PCA)
 * 工作原理：在灯条附近建立 ROI，进行加权质心和主成分分析，通过亮度跳变点精确定位灯条两端。
 */
void Detector::lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const
{
  // 1. 初始化常量
  constexpr float MAX_BRIGHTNESS = 25;  // 归一化参考亮度
  constexpr float ROI_SCALE = 0.07;     // ROI 扩边比例
  constexpr float SEARCH_START = 0.4;   // 亮度搜索起始位置
  constexpr float SEARCH_END = 0.6;     // 亮度搜索结束位置

  // 2. 建立局部 ROI 并进行归一化
  cv::Rect roi_box = lightbar.rotated_rect.boundingRect();
  roi_box.x -= roi_box.width * ROI_SCALE;
  roi_box.y -= roi_box.height * ROI_SCALE;
  roi_box.width += 2 * roi_box.width * ROI_SCALE;
  roi_box.height += 2 * roi_box.height * ROI_SCALE;
  roi_box &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);

  cv::Mat roi = gray_img(roi_box);
  const float mean_val = cv::mean(roi)[0];
  roi.convertTo(roi, CV_32F);
  cv::normalize(roi, roi, 0, MAX_BRIGHTNESS, cv::NORM_MINMAX);

  // 3. 计算 ROI 内的物理质心
  const cv::Moments moments = cv::moments(roi);
  const cv::Point2f centroid(
    moments.m10 / moments.m00 + roi_box.x, moments.m01 / moments.m00 + roi_box.y);

  // 4. 提取拟合点云并进行 PCA 计算主轴方向
  std::vector<cv::Point2f> points;
  for (int i = 0; i < roi.rows; ++i) {
    for (int j = 0; j < roi.cols; ++j) {
      if (roi.at<float>(i, j) > 1e-3) points.emplace_back(j, i);
    }
  }

  cv::PCA pca(cv::Mat(points).reshape(1), cv::Mat(), cv::PCA::DATA_AS_ROW);
  cv::Point2f axis(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
  axis /= cv::norm(axis);
  if (axis.y > 0) axis = -axis; // 强制轴指向上方

  // 5. 内部闭包：沿轴搜索亮度最剧烈跳变的点 (Edge detection on axis)
  const auto find_corner = [&](int direction) -> cv::Point2f {
    const float dx = axis.x * direction;
    const float dy = axis.y * direction;
    const float search_length = lightbar.length * (SEARCH_END - SEARCH_START);

    std::vector<cv::Point2f> candidates;
    const int half_width = (lightbar.width - 2) / 2;
    for (int i_offset = -half_width; i_offset <= half_width; ++i_offset) {
      cv::Point2f start_point(
        centroid.x + lightbar.length * SEARCH_START * dx + i_offset,
        centroid.y + lightbar.length * SEARCH_START * dy);

      cv::Point2f corner = start_point;
      float max_diff = 0;
      bool found = false;
      for (float step = 0; step < search_length; ++step) {
        const cv::Point2f cur_point(start_point.x + dx * step, start_point.y + dy * step);
        if (cur_point.x < 0 || cur_point.x >= gray_img.cols || cur_point.y < 0 || cur_point.y >= gray_img.rows) break;

        const auto prev_val = gray_img.at<uchar>(cv::Point2i(cur_point - cv::Point2f(dx, dy)));
        const auto cur_val = gray_img.at<uchar>(cv::Point2i(cur_point));
        const float diff = prev_val - cur_val;

        if (diff > max_diff && prev_val > mean_val) {
          max_diff = diff;
          corner = cur_point - cv::Point2f(dx, dy);
          found = true;
        }
      }
      if (found) candidates.push_back(corner);
    }
    return candidates.empty() ? cv::Point2f(-1, -1) : std::accumulate(candidates.begin(), candidates.end(), cv::Point2f(0, 0)) / static_cast<float>(candidates.size());
  };

  // 6. 更新校正后的灯条顶点
  lightbar.top = find_corner(1);
  lightbar.bottom = find_corner(-1);
}

}  // namespace auto_aim