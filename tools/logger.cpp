#include "logger.cpp"

#include <fmt/chrono.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace tools
{

// 全局静态 logger 变量
std::shared_ptr<spdlog::logger> logger_ = nullptr;

/**
 * @brief 内部初始化日志配置
 * 逻辑：
 * 1. 生成基于当前时间戳的日志文件名（例如：logs/2026-04-19_13-00-00.log）。
 * 2. 配置文件输出流 (File Sink) 和 控制台输出流 (Console Sink)。
 * 3. 设置默认日志等级为 DEBUG。
 * 4. 设置 flush 策略，确保重要信息 (INFO 及以上) 即时落地。
 */
void set_logger()
{
  auto now = std::chrono::system_clock::now();
  auto file_name = fmt::format("logs/{:%Y-%m-%d_%H-%M-%S}.log", now);
  
  // 1. 创建多线程安全的文件输出槽
  auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name, true);
  file_sink->set_level(spdlog::level::debug);

  // 2. 创建彩色终端输出槽
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::debug);

  // 3. 构建组合 Logger 并注册
  logger_ = std::make_shared<spdlog::logger>("", spdlog::sinks_init_list{file_sink, console_sink});
  logger_->set_level(spdlog::level::debug);
  
  // 4. 开启即时冲刷 (用于事故回溯)
  logger_->flush_on(spdlog::level::info);
}

/**
 * @brief 日志管理单例接口
 */
std::shared_ptr<spdlog::logger> logger()
{
  // 懒加载模式初始化
  if (!logger_) set_logger();
  return logger_;
}

}  // namespace tools
