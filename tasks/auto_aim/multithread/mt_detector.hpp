#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();
  //这里的结构化绑定
  //三个变量类型分别是：
"""
img -> cv::Mat
armors -> std::list<Armor>
t -> std::chrono::steady_clock::time_point

"""

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  YOLO yolo_;

  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>,true>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

"""
queue_ 为什么你觉得“没定义”
因为它不是 QYG_hero.cpp 里的局部变量，而是 MultiThreadDetector 类的私有成员变量，定义在类里面：
"""

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP