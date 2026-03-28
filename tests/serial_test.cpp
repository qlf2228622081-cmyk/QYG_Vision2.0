#include "io/gimbal/gimbal.hpp"
#include "tools/logger.hpp"


static bool init_serial(serial::Serial & serial, const std::string & port, uint32_t baud)
{
  try {
    serial.setPort(port);
    serial.setBaudrate(baud);
    serial.setFlowcontrol(serial::flowcontrol_none);
    serial.setParity(serial::parity_none);
    serial.setStopbits(serial::stopbits_one);
    serial.setBytesize(serial::eightbits);
    serial::Timeout time_out = serial::Timeout::simpleTimeout(200);  // 200ms 超时
    serial.setTimeout(time_out);
    serial.open();
    usleep(200000);  // 稍等串口稳定
    tools::logger()->info("serial port opened: {} @{}", port, baud);
    return true;
  } catch (const std::exception & e) {
    tools::logger()->warn("failed to open serial port {}: {}", port, e.what());
    return false;
  }
}

int main(int argc, char * argv[])
{
  const std::string port = "/dev/ttyUSB4";
  const uint32_t baud = 921600;

  serial::Serial serial;
  if (!init_serial(serial, port, baud)) return 1;

  uint8_t buffer[11] = {0};
  float q[4] = {0.0f};

  while (true)
  {
    auto read_bytes = serial.read(buffer, sizeof(buffer));
    if (read_bytes == 0) {
      tools::logger()->warn("No data received within timeout");
      continue; // 可按需决定是否退出
    }
    if(buffer[0] == 0x55 && buffer[1] == 0x59 ) {
      // if(buffer[10] != buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] +
      //                 buffer[5] + buffer[6] + buffer[7] + buffer[8] + buffer[9]) {
      //   tools::logger()->warn("Checksum error");
      //   continue;
      // }
      // Treat components as signed 16-bit to preserve original sign
      q[0] = static_cast<short>((static_cast<short>(buffer[3]) << 8) | buffer[2]) / 32768.0f;
      q[1] = static_cast<short>((static_cast<short>(buffer[5]) << 8) | buffer[4]) / 32768.0f;
      q[2] = static_cast<short>((static_cast<short>(buffer[7]) << 8) | buffer[6]) / 32768.0f;
      q[3] = static_cast<short>((static_cast<short>(buffer[9]) << 8) | buffer[8]) / 32768.0f;
      tools::logger()->info("q: {:+.4f} {:+.4f} {:+.4f} {:+.4f} norm: {:+.4f}", q[0], q[1], q[2], q[3],sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]));
    }
    else {
      tools::logger()->warn("Invalid header: {:02X} {:02X}", buffer[0], buffer[1]);
    }
  }
  
  return 0;
}