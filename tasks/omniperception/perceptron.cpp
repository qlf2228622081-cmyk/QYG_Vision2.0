#include "perceptron.cpp"

#include <chrono>
#include <memory>
#include <thread>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

namespace omniperception
{

/**
 * @brief Perceptron 构造函数
 * 逻辑：为 4 路 USB 工业相机分别初始化 YOLO 模型实例，并启动对应的感知后台线程。
 */
Perceptron::Perceptron(
  io::USBCamera * usbcam1, io::USBCamera * usbcam2, io::USBCamera * usbcam3,
  io::USBCamera * usbcam4, const std::string & config_path)
: detection_queue_(10), decider_(config_path), stop_flag_(false)
{
  // 1. 初始化各线程独立的推理模型 (false 参数通常表示不加载 GUI/调试功能)
  yolo_parallel1_ = std::make_shared<auto_aim::YOLO>(config_path, false);
  yolo_parallel2_ = std::make_shared<auto_aim::YOLO>(config_path, false);
  yolo_parallel3_ = std::make_shared<auto_aim::YOLO>(config_path, false);
  yolo_parallel4_ = std::make_shared<auto_aim::YOLO>(config_path, false);

  // 给模型加载预留一点时间
  std::this_thread::sleep_for(std::chrono::seconds(2));
  
  // 2. 启动四个感知子线程
  threads_.emplace_back([&] { parallel_infer(usbcam1, yolo_parallel1_); });
  threads_.emplace_back([&] { parallel_infer(usbcam2, yolo_parallel2_); });
  threads_.emplace_back([&] { parallel_infer(usbcam3, yolo_parallel3_); });
  threads_.emplace_back([&] { parallel_infer(usbcam4, yolo_parallel4_); });

  tools::logger()->info("Perceptron initialized with 4 parallel cameras.");
}

/**
 * @brief 析构函数
 * 逻辑：设置停止标志并等待各线程退出。
 */
Perceptron::~Perceptron()
{
  {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_flag_ = true;
  }
  condition_.notify_all();

  for (auto & t : threads_) {
    if (t.joinable()) t.join();
  }
  tools::logger()->info("Perceptron destructed gracefully.");
}

/**
 * @brief 提取汇总队列中的结果
 */
std::vector<DetectionResult> Perceptron::get_detection_queue()
{
  std::vector<DetectionResult> result;
  DetectionResult temp;

  // 将当前队列中积攒的所有识别结果一次性取出
  while (!detection_queue_.empty()) {
    detection_queue_.pop(temp);
    result.push_back(std::move(temp));
  }

  return result;
}

/**
 * @brief 并行推理子线程逻辑 (Worker)
 * 逻辑：死循环读取相机帧 -> 模型推理 -> 计算偏差角 -> 压入结果队列。
 */
void Perceptron::parallel_infer(
  io::USBCamera * cam, std::shared_ptr<auto_aim::YOLO> & yolov8_parallel)
{
  if (!cam) {
    tools::logger()->error("Camera pointer is null!");
    return;
  }
  try {
    while (true) {
      cv::Mat usb_img;
      std::chrono::steady_clock::time_point ts;

      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_flag_) break;
      }

      // 1. 读取图像 (含曝光与采集延时)
      cam->read(usb_img, ts);
      if (usb_img.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        continue;
      }

      // 2. 深度学习检测
      auto armors = yolov8_parallel->detect(usb_img);
      
      // 3. 如果在该相机视野内发现了目标
      if (!armors.empty()) {
        // 计算目标在“相机视野内”的偏离角度
        auto delta_angle = decider_.delta_angle(armors, cam->device_name);

        DetectionResult dr;
        dr.armors = std::move(armors);
        dr.timestamp = ts;
        // 将结果转为弧度并在结果结构中记录
        dr.delta_yaw = delta_angle[0] / 57.3;
        dr.delta_pitch = delta_angle[1] / 57.3;
        
        // 推送到主线程共享的队列中
        detection_queue_.push(dr);
      }
    }
  } catch (const std::exception & e) {
    tools::logger()->error("Exception in parallel_infer thread: {}", e.what());
  }
}

}  // namespace omniperception
