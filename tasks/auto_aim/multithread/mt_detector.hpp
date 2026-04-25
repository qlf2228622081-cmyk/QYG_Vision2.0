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

/**
 * @brief 多线程/异步检测器类 (MultiThreadDetector)
 * 逻辑：利用 OpenVINO 的异步推理接口（Async API），将图像预处理与硬件推理过程并行化。
 * 实现“生产者-消费者”模型，主线程 push 任务，工作线程（或硬件）异步计算，随后通过 pop 获取结果。
 */
class MultiThreadDetector
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式
   */
  MultiThreadDetector(const std::string & config_path, bool debug = false);

  /**
   * @brief 推送图片进行异步检测 (非阻塞)
   * @param img 待检测图像
   * @param t 图像捕获的时间戳 (用于时序补偿)
   */
  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  /**
   * @brief 弹出检测结果 (可能会阻塞等待推理完成)
   * @return tuple: <检测到的装甲板列表, 图像对应的时间戳>
   */
  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();

  /**
   * @brief 用于调试的结果弹出函数
   * @return tuple: <原始图像, 检测到的装甲板列表, 图像对应的时间戳>
   */
  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  /**
   * @brief 内部结果解析逻辑
   */
  std::list<Armor> pop_logic(
    cv::Mat & img, std::chrono::steady_clock::time_point & t, ov::InferRequest & infer_request);

  ov::Core core_;                    // OpenVINO 核心
  ov::CompiledModel compiled_model_; // 编译后的计算图
  std::string device_;               // 推理设备
  YOLO yolo_;                        // YOLO 后处理逻辑封装

  // 异步任务队列：存储 <图像, 时间戳, 推理请求句柄>
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>, true>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP