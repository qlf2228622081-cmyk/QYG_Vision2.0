#ifndef IO__USB2CAN_HPP
#define IO__USB2CAN_HPP

#include <linux/can.h>
#include <stdint.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>

#include "tools/logger.hpp"

#include "usb_fdcan.h"


using namespace std::chrono_literals;

namespace io
{
class USB2CAN
{
public:
  USB2CAN(const std::string & device_name, std::function<void(const can_frame & frame)> rx_handler, uint32_t bitrate)
  : device_name_(device_name),
    device_handle_(-1),
    rx_handler_(rx_handler),
    quit_(false),
    ok_(false),
    channel_(2),  // 默认使用通道1, 如果需要使用其他通道，可以修改这个值
    bitrate_(bitrate)  
  {
    try_open();

    // 守护线程
    daemon_thread_ = std::thread{[this] {
      while (!quit_) {
        std::this_thread::sleep_for(100ms);

        if (ok_) continue;

        if (read_thread_.joinable()) read_thread_.join();

        close();
        try_open();
      }
    }};
  }

  ~USB2CAN()
  {
    quit_ = true;
    if (daemon_thread_.joinable()) daemon_thread_.join();
    if (read_thread_.joinable()) read_thread_.join();
    close();
    tools::logger()->info("USB2CAN destructed.");
  }

  void write(can_frame * frame) const
  {
    if (device_handle_ < 0) {
      throw std::runtime_error("USB2CAN device not opened!");
    }

    FrameInfo info;
    // 提取 CAN ID（清除标志位）
    if (frame->can_id & CAN_EFF_FLAG) {
      info.canID = frame->can_id & CAN_EFF_MASK;  // 29位扩展帧ID
      info.frameType = EXTENDED;
    } else {
      info.canID = frame->can_id & CAN_SFF_MASK;  // 11位标准帧ID
      info.frameType = STANDARD;
    }
    info.dataLength = frame->can_dlc;

    int32_t ret = sendUSBCAN(device_handle_, channel_, &info, frame->data);
    if (ret != 8) {  // 8 表示成功
      throw std::runtime_error("USB2CAN send failed!");
    }
  }

  bool is_ok() const { return ok_; }

private:
  std::string device_name_;
  int32_t device_handle_;
  bool quit_;
  bool ok_;
  uint8_t channel_;
  std::thread read_thread_;
  std::thread daemon_thread_;
  std::function<void(const can_frame & frame)> rx_handler_;
  uint32_t bitrate_;

  void open()
  {
    device_handle_ = openUSBCAN(device_name_.c_str());
    if (device_handle_ < 0) {
      throw std::runtime_error("Failed to open USB2CAN device: " + device_name_);
    }

    // 将波特率（bps）转换为 SDK 枚举值
    uint8_t fdcan_speed;
    switch (bitrate_) {
      case 500000:
        fdcan_speed = FDCAN_500K;
        break;
      case 1000000:
        fdcan_speed = FDCAN_1M;
        break;
      case 2000000:
        fdcan_speed = FDCAN_2M;
        break;
      case 4000000:
        fdcan_speed = FDCAN_4M;
        break;
      default:
        throw std::runtime_error("Invalid bitrate: " + std::to_string(bitrate_) + 
                                 " bps. Supported values: 500000, 1000000, 2000000, 4000000");
    }
    // int32_t ret = configUSBCAN(device_handle_, channel_, CAN, fdcan_speed, fdcan_speed);  
    // if (ret != 0) {  
    //   throw std::runtime_error("Failed to configure USB2CAN device: " + device_name_);
    // }

    // 接收线程
    read_thread_ = std::thread([this]() {
      ok_ = true;
      while (!quit_) {
        std::this_thread::sleep_for(10us);

        try {
          read();
        } catch (const std::exception & e) {
          tools::logger()->warn("USB2CAN::read() failed: {}", e.what());
          ok_ = false;
          break;
        }
      }
    });

    tools::logger()->info("USB2CAN opened: {} with bitrate: {} bps", device_name_, bitrate_);
  }

  void try_open()
  {
    try {
      open();
    } catch (const std::exception & e) {
      tools::logger()->warn("USB2CAN::open() failed: {}", e.what());
    }
  }

  void read()
  {
    FrameInfo info;
    uint8_t data[8];
    uint8_t channel;
    int32_t timeout = 10000;  // 10ms timeout in microseconds

    int32_t ret = readUSBCAN(device_handle_, &channel, &info, data, timeout);
    if (ret == 0) {  // 0 表示成功
      can_frame frame;
      frame.can_id = info.canID;
      if (info.frameType == EXTENDED) {
        frame.can_id |= CAN_EFF_FLAG;
      }
      frame.can_dlc = info.dataLength;
      std::memcpy(frame.data, data, 8);

      rx_handler_(frame);
    }
  }

  void close()
  {
    if (device_handle_ < 0) return;
    closeUSBCAN(device_handle_);
    device_handle_ = -1;
  }
};

}  // namespace io

#endif  // IO__USB2CAN_HPP

