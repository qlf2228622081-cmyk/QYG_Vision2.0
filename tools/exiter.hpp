#ifndef TOOLS__EXITER_HPP
#define TOOLS__EXITER_HPP

namespace tools
{

/**
 * @brief 程序退出管理器 (Exiter)
 * 逻辑：用于优雅地中断程序运行。它通过捕获系统信号（如 Ctrl+C 触发的 SIGINT），
 * 将内部标志位设为 true，从而通知各个模块的安全循环（While Loop）退出。
 */
class Exiter
{
public:
  /**
   * @brief 构造函数：注册信号处理函数
   */
  Exiter();

  /**
   * @brief 轮询接口：检查当前是否已触发退出信号
   */
  bool exit() const;
};

}  // namespace tools

#endif  // TOOLS__EXITER_HPP