#ifndef TOOLS__CRC_HPP
#define TOOLS__CRC_HPP

#include <cstdint>

namespace tools
{

/**
 * @brief 数据校验工具组 (CRC)
 * 逻辑：提供循环冗余校验 (CRC8/CRC16) 的计算与校验功能，常用于串口通信协议中，
 * 确保机器视觉系统与底层电控板之间数据传输的准确性。
 */

/**
 * @brief 计算 CRC8 校验码
 * @param data 待计算数据指针
 * @param len 数据长度 (不含校验位本身)
 */
uint8_t get_crc8(const uint8_t * data, uint16_t len);

/**
 * @brief 执行 CRC8 完整性检查
 * @param len 包含校验位在内的总长度 (校验位应在末尾)
 */
bool check_crc8(const uint8_t * data, uint16_t len);

/**
 * @brief 计算 CRC16 校验码
 */
uint16_t get_crc16(const uint8_t * data, uint32_t len);

/**
 * @brief 执行 CRC16 完整性检查 (校验位按小端序存储)
 */
bool check_crc16(const uint8_t * data, uint32_t len);

}  // namespace tools

#endif  // TOOLS__CRC_HPP
