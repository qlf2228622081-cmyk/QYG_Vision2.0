#ifndef TOOLS__YAML_HPP
#define TOOLS__YAML_HPP

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"

namespace tools
{

/**
 * @brief YAML 配置文件加载工具
 * 逻辑：安全地加载本地 .yaml 配置文件。如果文件缺失或格式错误，
 * 会立即通过 Logger 报错并强制终止程序，防止系统在非法参数下运行。
 */
inline YAML::Node load(const std::string & path)
{
  try {
    return YAML::LoadFile(path);
  } catch (const YAML::BadFile & e) {
    logger()->error("[YAML] 无法加载配置文件，请检查路径: {}", e.what());
    exit(1);
  } catch (const YAML::ParserException & e) {
    logger()->error("[YAML] 语法错误: {}", e.what());
    exit(1);
  }
}

/**
 * @brief 类型安全的配置项读取接口
 * 逻辑：如果指定的 Key 不存在，会报错并退出程序。
 */
template <typename T>
inline T read(const YAML::Node & yaml, const std::string & key)
{
  if (yaml[key]) return yaml[key].as<T>();
  logger()->error("[YAML] 配置项 '{}' 缺失，请检查 .yaml 文件！", key);
  exit(1);
}

}  // namespace tools

#endif  // TOOLS__YAML_HPP