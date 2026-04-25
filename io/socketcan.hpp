#ifndef IO__SOCKETCAN_HPP
#define IO__SOCKETCAN_HPP

#include <linux/can.h>
#include <net/if.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <thread>

#include "tools/logger.hpp"

using namespace std::chrono_literals;

// Epoll 同时处理的最大事件数
constexpr int MAX_EVENTS = 10;

namespace io
{

/**
 * @brief SocketCAN 驱动类 (SocketCAN)
 * 逻辑：基于 Linux 原生 SocketCAN 接口实现的 CAN 总线驱动。
 * 特性：
 * 1. 采用 epoll I/O 多路复用技术，实现高频率（>1kHz）下的高效、低延迟数据接收。
 * 2. 具备“守护线程 (Daemon Thread)”机制，当 CAN 接口（如 candump/can0）掉线或异常时，自动重新尝试绑定。
 * 3. 异步读取设计，通过回调函数 (rx_handler) 将解析任务解耦。
 */
class SocketCAN
{
public:
  /**
   * @brief 构造函数：建立连接并启动接收后台
   * @param interface 网卡名称 (如 "can0", "vcan0")
   * @param rx_handler 接收到数据帧时的处理回调函数
   */
  SocketCAN(const std::string & interface, std::function<void(const can_frame & frame)> rx_handler)
  : interface_(interface),
    socket_fd_(-1),
    epoll_fd_(-1),
    rx_handler_(rx_handler),
    quit_(false),
    ok_(false)
  {
    try_open();

    // 守护线程逻辑：每 100ms 检查连接健康度 ok_，若异常则安全关闭并重连。
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

  /**
   * @brief 析构实现：安全释放所有 fd 和线程
   */
  ~SocketCAN()
  {
    quit_ = true;
    if (daemon_thread_.joinable()) daemon_thread_.join();
    if (read_thread_.joinable()) read_thread_.join();
    close();
    tools::logger()->info("SocketCAN 驱动已安全释放。");
  }

  /**
   * @brief 发送 CAN 数据帧
   * @param frame 待发送的 can_frame 结构体
   */
  void write(can_frame * frame) const
  {
    // 利用 Linux 原生 write 系统调用
    if (::write(socket_fd_, frame, sizeof(can_frame)) == -1)
      throw std::runtime_error("SocketCAN 写入失败，请检查总线状态！");
  }

private:
  std::string interface_;
  int socket_fd_;                       // RAW_CAN 套接字描述符
  int epoll_fd_;                        // Epoll 实例描述符
  bool quit_;                           // 退出标志
  bool ok_;                             // 连接健康标志
  std::thread read_thread_;             // 异步读取线程
  std::thread daemon_thread_;          // 守护重连线程
  can_frame frame_;                    // 接收暂存区
  epoll_event events_[MAX_EVENTS];    // Epoll 事件集
  std::function<void(const can_frame & frame)> rx_handler_;

  /**
   * @brief 底层 Socket 建立与 Epoll 绑定
   */
  void open()
  {
    // 1. 创建原生 SocketCAN 套接字
    socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) throw std::runtime_error("无法创建 CAN Socket！");

    // 2. 获取网卡索引
    ifreq ifr;
    std::strncpy(ifr.ifr_name, interface_.c_str(), IFNAMSIZ - 1);
    if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0)
      throw std::runtime_error("获取接口索引失败，请检查网卡是否存在！");

    // 3. 绑定 socket 到特定网卡
    sockaddr_can addr;
    std::memset(&addr, 0, sizeof(sockaddr_can));
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(socket_fd_, (sockaddr *)&addr, sizeof(sockaddr_can)) < 0) {
      ::close(socket_fd_);
      throw std::runtime_error("绑定接口失败！");
    }

    // 4. 配置 Epoll 监听 (非阻塞读取)
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ == -1) throw std::runtime_error("无法创建 epoll fd！");

    epoll_event ev;
    ev.events = EPOLLIN; // 监听可读状态
    ev.data.fd = socket_fd_;
    if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, ev.data.fd, &ev))
      throw std::runtime_error("无法将 socket 添加入 epoll！");

    // 5. 启动读取子线程
    read_thread_ = std::thread([this]() {
      ok_ = true;
      while (!quit_) {
        // 微小休眠防止死循环撑爆 CPU，平衡实时性
        std::this_thread::sleep_for(10us);
        try {
          read(); // 执行 epoll_wait 与 recv
        } catch (const std::exception & e) {
          tools::logger()->warn("SocketCAN 读取过程中断: {}", e.what());
          ok_ = false; // 触发守护线程重启
          break;
        }
      }
    });

    tools::logger()->info("SocketCAN [{}] 已建立连接。", interface_);
  }

  void try_open()
  {
    try {
      open();
    } catch (const std::exception & e) {
      tools::logger()->warn("SocketCAN 开启提示: {}", e.what());
    }
  }

  /**
   * @brief 数据接收主逻辑
   */
  void read()
  {
    // 等待事件触发 (2ms 超时)
    int num_events = epoll_wait(epoll_fd_, events_, MAX_EVENTS, 2);
    if (num_events == -1) throw std::runtime_error("epoll_wait 超时或异常！");

    for (int i = 0; i < num_events; i++) {
      // 执行非阻塞接收数据
      ssize_t num_bytes = recv(socket_fd_, &frame_, sizeof(can_frame), MSG_DONTWAIT);
      if (num_bytes == -1) throw std::runtime_error("CAN 数据包接收异常！");

      // 分发给上层处理
      rx_handler_(frame_);
    }
  }

  /**
   * @brief 资源回收
   */
  void close()
  {
    if (socket_fd_ == -1) return;
    epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, socket_fd_, NULL);
    ::close(epoll_fd_);
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
};

}  // namespace io

#endif  // IO__SOCKETCAN_HPP