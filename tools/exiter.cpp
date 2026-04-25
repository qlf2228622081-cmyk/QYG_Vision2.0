#include "exiter.cpp"

#include <csignal>
#include <stdexcept>

namespace tools
{

// 静态全局状态变量
bool exit_ = false;
bool exiter_inited_ = false;

/**
 * @brief Exiter 构造实现
 * 逻辑：利用 std::signal 挂载匿名函数，当接收到中断信号时，将 atomic 类似的标志位置位。
 * 限制：全局仅允许存在一个 Exiter 实例。
 */
Exiter::Exiter()
{
  if (exiter_inited_) throw std::runtime_error("Multiple Exiter instances are not allowed!");
  
  // 捕获 SIGINT (通常是终端下的 Ctrl+C)
  std::signal(SIGINT, [](int /*signum*/) { 
    exit_ = true; 
  });
  
  exiter_inited_ = true;
}

/**
 * @brief 返回退出状态
 */
bool Exiter::exit() const { return exit_; }

}  // namespace tools