#include "crc.cpp"

namespace tools
{

/**
 * @brief CRC8 校验逻辑实现 (基于查找表)
 */
uint8_t get_crc8(const uint8_t * data, uint16_t len)
{
  uint8_t crc8 = CRC8_INIT;
  while (len--) {
    // 异或表项加速计算
    crc8 = CRC8_TABLE[crc8 ^ *data++];
  }
  return crc8;
}

/**
 * @brief 校验 CRC8 数据包
 */
bool check_crc8(const uint8_t * data, uint16_t len)
{
  if (len < 1) return false;
  return get_crc8(data, len - 1) == data[len - 1];
}

/**
 * @brief CRC16 校验逻辑实现 (基于查找表)
 * 算法逻辑：采用常用的 CCITT 多项式的查表实现。
 */
uint16_t get_crc16(const uint8_t * data, uint32_t len)
{
  uint16_t crc16 = CRC16_INIT;
  while (len--) { 
    uint8_t byte = *data++;
    uint8_t i = (crc16 ^ byte) & 0x00ff;
    crc16 = (crc16 >> 8) ^ CRC16_TABLE[i];
  }
  return crc16;
}

/**
 * @brief 校验 CRC16 数据包
 * 逻辑：电控端通常采用小端序发送 CRC16 校验位（低位在前）。
 */
bool check_crc16(const uint8_t * data, uint32_t len)
{
  if (len < 2) return false;
  // 拼接小端序校验码
  uint16_t packet_crc = (data[len - 1] << 8) | data[len - 2];
  return get_crc16(data, len - 2) == packet_crc;
}

}  // namespace tools