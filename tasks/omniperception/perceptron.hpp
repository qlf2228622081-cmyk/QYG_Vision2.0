#ifndef OMNIPERCEPTION__PERCEPTRON_HPP
#define OMNIPERCEPTION__PERCEPTRON_HPP

#include <chrono>
#include <list>
#include <memory>

#include "decider.hpp"
#include "detection.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/thread_pool.hpp"
#include "tools/thread_safe_queue.hpp"

namespace omniperception
{

/**
 * @brief 全场感知引擎 (Perceptron)
 * 逻辑：负责管理多个工业相机的并行驱动与异步推理。
 * 它为每一个接入的相机创建一个独立的子线程，每个子线程独立进行“图像读取-模型推理-初级过滤”，
 * 并将结果汇总到一个线程安全的感知结果队列中。
 */
class Perceptron
{
public:
  /**
   * @brief 构造函数：初始化多路相机感知线程
   * @param usbcams 指向接入的多个相机对象的指针
   * @param config_path 配置文件路径
   */
  Perceptron(
    io::USBCamera * usbcma1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
    io::USBCamera * usbcam4, const std::string & config_path);

  /**
   * @brief 析构函数：优雅地停止所有感知线程
   */
  ~Perceptron();

  /**
   * @brief 获取当前已堆积的所有感知结果
   * 从线程安全队列中取出所有 DetectionResult 并移交给调用者。
   */
  std::vector<DetectionResult> get_detection_queue();

  /**
   * @brief 线程子任务：单个相机的并行推理生命周期
   */
  void parallel_infer(io::USBCamera * cam, std::shared_ptr<auto_aim::YOLO> & yolo_parallel);

private:
  std::vector<std::thread> threads_;                   // 线程池池（手动管理）
  tools::ThreadSafeQueue<DetectionResult> detection_queue_; // 线程安全的结果汇总队列

  // 每个线程独立的推理模型实例，避免线程竞争
  std::shared_ptr<auto_aim::YOLO> yolo_parallel1_;
  std::shared_ptr<auto_aim::YOLO> yolo_parallel2_;
  std::shared_ptr<auto_aim::YOLO> yolo_parallel3_;
  std::shared_ptr<auto_aim::YOLO> yolo_parallel4_;

  Decider decider_;        // 关联决策器 (主要用于计算 delta_angle)
  bool stop_flag_;         // 线程停止标志位
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

}  // namespace omniperception
#endif