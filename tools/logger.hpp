#ifndef TOOLS__LOGGER_HPP
#define TOOLS__LOGGER_HPP

#include <spdlog/spdlog.h>

namespace tools
{

/**
 * @brief 获取全局日志记录器句柄
 * 逻辑：单例模式或静态全局初始化，返回一个共享的 spdlog 实例。
 * 能够同时向控制台 (stdout) 和 本地文件 (logs/目录) 输出信息。
 */
std::shared_ptr<spdlog::logger> logger();

}  // namespace tools

#endif  // TOOLS__LOGGER_HPP