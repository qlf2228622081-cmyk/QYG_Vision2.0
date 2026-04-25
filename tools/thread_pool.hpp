#ifndef TOOLS__THREAD_POOL_HPP
#define TOOLS__THREAD_POOL_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"

namespace tools
{

/**
 * @brief 图像帧数据结构 (Frame)
 * 用于多线程流水线中传递图像及其关联元数据（时间戳、云台位姿等）。
 */
struct Frame
{
  int id;                                       // 帧序号 (用于时序同步)
  cv::Mat img;                                  // 原始图像
  std::chrono::steady_clock::time_point t;      // 采集时刻时间戳
  Eigen::Quaterniond q;                         // 采集时刻云台四元数 (从电控获取)
  std::list<auto_aim::Armor> armors;            // 识别出的装甲板列表
};

/**
 * @brief 顺序队列 (OrderedQueue)
 * 逻辑：为了解决多线程并行推理导致的“乱序返回”问题。
 * 它内部维护一个缓冲区，确保输出的帧序号（Frame ID）是严格递增连续的。
 */
class OrderedQueue
{
public:
  OrderedQueue() : current_id_(1) {}
  ~OrderedQueue();

  /**
   * @brief 将处理完的帧压入队列
   * 如果该 ID 正好是预期的下一个 ID，则通过并检查缓冲区；
   * 否则将其暂存在缓冲区中等待前面的 ID 到达。
   */
  void enqueue(const tools::Frame & item);

  /**
   * @brief 阻塞式获取下一帧 (严格保证顺序)
   */
  tools::Frame dequeue();

  /**
   * @brief 尝试非阻塞获取下一帧
   */
  bool try_dequeue(tools::Frame & item);

  size_t get_size() { return main_queue_.size() + buffer_.size(); }

private:
  std::queue<tools::Frame> main_queue_;          // 已就绪的顺序队列
  std::unordered_map<int, tools::Frame> buffer_; // 暂存乱序到达的帧
  int current_id_;                               // 当前期望的下一个 ID
  std::mutex mutex_;
  std::condition_variable cond_var_;
};

/**
 * @brief 通用任务线程池 (ThreadPool)
 * 逻辑：标准的固定线程数量生产者-消费者模型，用于执行异步计算任务。
 */
class ThreadPool
{
public:
  /**
   * @brief 创建指定数量的工作线程
   */
  ThreadPool(size_t num_threads) : stop(false)
  {
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // 等待直到队列不为空或线程池停止
            condition.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task(); // 执行具体逻辑
        }
      });
    }
  }

  /**
   * @brief 停止所有线程并清理未完成任务
   */
  ~ThreadPool();

  /**
   * @brief 向池中投放新任务
   */
  template <class F>
  void enqueue(F && f)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
      tasks.emplace(std::forward<F>(f));
    }
    condition.notify_one();
  }

private:
  std::vector<std::thread> workers;         
  std::queue<std::function<void()>> tasks;  
  std::mutex queue_mutex;                   
  std::condition_variable condition;        
  bool stop;                                
};
}  // namespace tools

#endif  // TOOLS__THREAD_POOL_HPP
