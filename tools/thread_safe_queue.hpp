#ifndef TOOLS__THREAD_SAFE_QUEUE_HPP
#define TOOLS__THREAD_SAFE_QUEUE_HPP

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>

namespace tools
{

/**
 * @brief 线程安全队列 (ThreadSafeQueue)
 * 逻辑：基于互斥锁和条件变量封装的线程安全模板。
 * 支持固定容量限制，当队列满时可选：丢弃旧数据（PopWhenFull=true）或调用自定义处理回调。
 */
template <typename T, bool PopWhenFull = false>
class ThreadSafeQueue
{
public:
  /**
   * @brief 构造函数
   * @param max_size 队列最大允许积压长度
   * @param full_handler 当队列满时的外部处理逻辑
   */
  ThreadSafeQueue(
    size_t max_size, std::function<void(void)> full_handler = [] {})
  : max_size_(max_size), full_handler_(full_handler)
  {
  }

  /**
   * @brief 入队 (Push) 
   * 如果队列已满，根据策略处理 (默认返回或弹出头部)
   */
  void push(const T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.size() >= max_size_) {
      if (PopWhenFull) {
        queue_.pop(); // 弹出最旧的数据，腾出空间
      } else {
        full_handler_(); // 调用溢出处理
        return;
      }
    }

    queue_.push(value);
    not_empty_condition_.notify_all(); // 通知等待中的消费者
  }

  /**
   * @brief 阻塞式弹出 (Pop)
   * 引用传参，确保在重负载下减少不必要的拷贝开销。
   */
  void pop(T & value)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // 等待队列非空
    not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });

    if (queue_.empty()) return;

    value = std::move(queue_.front());
    queue_.pop();
  }

  /**
   * @brief 获取队首元素 (阻塞并返回副本)
   */
  T pop()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  /**
   * @brief 返回队首元素引用 (阻塞直到有数据)
   */
  T front()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_condition_.wait(lock, [this] { return !queue_.empty(); });
    return queue_.front();
  }

  /**
   * @brief 判断是否为空 (线程安全)
   */
  bool empty()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  /**
   * @brief 清空队列
   */
  void clear()
  {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!queue_.empty()) queue_.pop();
    not_empty_condition_.notify_all();
  }

private:
  std::queue<T> queue_;
  size_t max_size_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_condition_;
  std::function<void(void)> full_handler_;
};

}  // namespace tools

#endif  // TOOLS__THREAD_SAFE_QUEUE_HPP