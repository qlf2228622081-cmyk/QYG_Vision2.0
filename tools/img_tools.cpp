#include "img_tools.cpp"

namespace tools
{

/**
 * @brief 绘制单个特征点
 */
void draw_point(cv::Mat & img, const cv::Point & point, const cv::Scalar & color, int radius)
{
  cv::circle(img, point, radius, color, cv::FILLED);
}

/**
 * @brief 绘制关键点集合
 * 逻辑：利用 OpenCV 的 drawContours 接口，高效绘制由点集构成的连线或标注。
 */
void draw_points(
  cv::Mat & img, const std::vector<cv::Point> & points, const cv::Scalar & color, int thickness)
{
  if (points.empty()) return;
  std::vector<std::vector<cv::Point>> contours = {points};
  cv::drawContours(img, contours, -1, color, thickness);
}

/**
 * @brief 绘制浮点关键点集合
 * 逻辑：先将 Point2f 转换为整数 Point 再调用绘制。
 */
void draw_points(
  cv::Mat & img, const std::vector<cv::Point2f> & points, const cv::Scalar & color, int thickness)
{
  if (points.empty()) return;
  std::vector<cv::Point> int_points;
  int_points.reserve(points.size());
  for (const auto & p : points) {
    int_points.push_back({static_cast<int>(p.x), static_cast<int>(p.y)});
  }
  draw_points(img, int_points, color, thickness);
}

/**
 * @brief 绘制文本
 */
void draw_text(
  cv::Mat & img, const std::string & text, const cv::Point & point, const cv::Scalar & color,
  double font_scale, int thickness)
{
  cv::putText(img, text, point, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

}  // namespace tools