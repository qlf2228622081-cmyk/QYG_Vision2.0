#include "cboard.cpp"

#include <net/if.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io {

/**
 * @brief CBoard 构造函数
 * 逻辑：连接 SocketCAN 虚拟或硬件接口，并初始化 IMU 接收队列。
 */
CBoard::CBoard(const std::string &config_path)
    : mode(Mode::idle), shoot_mode(ShootMode::left_shoot), bullet_speed(0),
      enemy_color_(EnemyColor::red), queue_(5000), socketcan_(nullptr) {
  
  auto yaml = tools::load(config_path);
  quaternion_canid_ = tools::read<int>(yaml, "quaternion_canid");
  bullet_speed_canid_ = tools::read<int>(yaml, "bullet_speed_canid");
  send_canid_ = tools::read<int>(yaml, "send_canid");

  std::string can_interface = tools::read<std::string>(yaml, "can_interface");

  // 1. 系统可用性检查
  if (!check_socketcan_available(can_interface)) {
    throw std::runtime_error("SocketCAN 接口不可用: " + can_interface);
  }

  // 2. 初始化 CAN 驱动并绑定回调
  try {
    socketcan_ = std::make_unique<SocketCAN>(
        can_interface,
        std::bind(&CBoard::callback, this, std::placeholders::_1));
    tools::logger()->info("[CBoard] 已连接到 SocketCAN 接口: {}", can_interface);
  } catch (const std::exception &e) {
    throw std::runtime_error("SocketCAN 初始化失败: " + std::string(e.what()));
  }

  // 3. 阻塞等待最初的几帧 IMU 数据，确保插值算法有初值
  tools::logger()->info("[Cboard] 正在等待 IMU 初始数据...");
  queue_.pop(data_ahead_);
  queue_.pop(data_behind_);
  tools::logger()->info("[Cboard] 通信接口已就绪.");
}

/**
 * @brief 插值获取指定时刻的位姿 (Core Logic)
 * 算法逻辑：
 * 1. 在 history queue 中线性搜索，找到时间戳 timestamp 刚好处于 data_ahead 和 data_behind 之间。
 * 2. 如果队列为空，则进入阻塞等待并提醒用户检查硬件连接。
 * 3. 执行球面线性插值 (Slerp)，返回精确的归一化四元数。
 */
Eigen::Quaterniond
CBoard::imu_at(std::chrono::steady_clock::time_point timestamp) {
  // 维护搜索窗口：如果后一帧还比目标时间早，则向前推进前一帧
  if (data_behind_.timestamp < timestamp)
    data_ahead_ = data_behind_;

  while (true) {
    if (queue_.empty()) {
      auto last_log_time = std::chrono::steady_clock::now();
      while (queue_.empty()) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_log_time >= std::chrono::seconds(1)) {
          tools::logger()->warn("[CBoard] IMU 队列为空，请检查电控 CAN 线连接！");
          last_log_time = now;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    queue_.pop(data_behind_);
    // 找到跨越该时间点的两个数据帧
    if (data_behind_.timestamp > timestamp)
      break;
    data_ahead_ = data_behind_;
  }

  // 准备插值参数
  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();
  
  double t_ab = std::chrono::duration<double>(data_behind_.timestamp - data_ahead_.timestamp).count();
  double t_ac = std::chrono::duration<double>(timestamp - data_ahead_.timestamp).count();

  // 执行 SLERP 插值
  double k = (t_ab > 1e-6) ? (t_ac / t_ab) : 0.0;
  return q_a.slerp(k, q_b).normalized();
}

/**
 * @brief 发送控制指令
 * 协议定义：
 * [0]: control_flag, [1]: shoot_flag
 * [2-3]: yaw (scaled by 1e4, big endian)
 * [4-5]: pitch (scaled by 1e4, big endian)
 * [6-7]: horizon_distance (scaled by 1e4, big endian)
 */
void CBoard::send(Command command) const {
  can_frame frame;
  frame.can_id = send_canid_;
  frame.can_dlc = 8;
  frame.data[0] = (command.control) ? 1 : 0;
  frame.data[1] = (command.shoot) ? 1 : 0;
  
  // 16位有符号整型转换，比例因子为 10000
  int16_t yaw_val = static_cast<int16_t>(command.yaw * 1e4);
  int16_t pitch_val = static_cast<int16_t>(command.pitch * 1e4);
  int16_t dist_val = static_cast<int16_t>(command.horizon_distance * 1e4);

  frame.data[2] = (yaw_val >> 8) & 0xFF;
  frame.data[3] = yaw_val & 0xFF;
  frame.data[4] = (pitch_val >> 8) & 0xFF;
  frame.data[5] = pitch_val & 0xFF;
  frame.data[6] = (dist_val >> 8) & 0xFF;
  frame.data[7] = dist_val & 0xFF;

  write_can_frame(frame);
}

/**
 * @brief CAN 接收回调处理
 * 逻辑：解析来自电控的不同帧 ID，提取四元数分量并压入队列。
 * 注意：此处目前根据 ID 粗略判定了敌方颜色（实际应用中应由专用位标志位判定）。
 */
void CBoard::callback(const can_frame &frame) {
  auto timestamp = std::chrono::steady_clock::now();

  // 解析四元数数据包
  if (frame.can_id == quaternion_canid_ || frame.can_id == bullet_speed_canid_) {
    // 假设不同的 ID 代表当前检测到的阵营颜色 (临时逻辑)
    enemy_color_.store((frame.can_id == quaternion_canid_) ? EnemyColor::red : EnemyColor::blue);
    
    // 协议：每个分量占用 16bit，还原为由 [-1, 1] 范围的浮点数
    auto w_raw = (uint16_t)(frame.data[0] << 8 | frame.data[1]);
    auto x_raw = (uint16_t)(frame.data[2] << 8 | frame.data[3]);
    auto y_raw = (uint16_t)(frame.data[4] << 8 | frame.data[5]);
    auto z_raw = (uint16_t)(frame.data[6] << 8 | frame.data[7]);

    double w = uint_to_float(w_raw, q_min, q_max, 16);
    double x = uint_to_float(x_raw, q_min, q_max, 16);
    double y = uint_to_float(y_raw, q_min, q_max, 16);
    double z = uint_to_float(z_raw, q_min, q_max, 16);

    // 有效性检查 (范数应接近 1)
    if (std::abs(w*w + x*x + y*y + z*z - 1.0) < 0.1) {
      queue_.push({{w, x, y, z}, timestamp});
    }
  }
}

/**
 * @brief 数据解码辅助：定点数转浮点数
 */
float CBoard::uint_to_float(int x_int, float x_min, float x_max, int bits) {
  float span = x_max - x_min;
  return ((float)x_int) * span / (float)((1 << bits) - 1) + x_min;
}

/**
 * @brief 底层接口探测
 */
bool CBoard::check_socketcan_available(const std::string &interface) {
  int sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (sock < 0) return false;
  ifreq ifr;
  std::strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
  bool available = (ioctl(sock, SIOCGIFINDEX, &ifr) >= 0);
  ::close(sock);
  return available;
}

} // namespace io