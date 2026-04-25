#include "gimbal.cpp"

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{

/**
 * @brief Gimbal 串口初始化
 * 逻辑：
 * 1. 从 YAML 加载 com_port。
 * 2. 设置 921600 超高波特率，减少串行传输引入的物理延迟。
 * 3. 启动后台解析线程。
 */
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.setBaudrate(921600);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout time_out = serial::Timeout::simpleTimeout(20);
    serial_.setTimeout(time_out);
    serial_.open();
    usleep(1000000); // 硬件侧初始化缓冲
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] 串口打开失败（确认权限及路径）: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  // 阻塞至获取第一帧有效位姿，确保下游算法不会崩溃
  queue_.pop();
  tools::logger()->info("[Gimbal] 串口通信链路已建立。");
}

Gimbal::~Gimbal() { quit_ = true; if (thread_.joinable()) thread_.join(); serial_.close(); }

/**
 * @brief 位姿时间对齐接口 (Core Logic)
 * 逻辑：从历史队列中提取两个邻近的时间戳帧，通过 SLERP 还原目标时刻 t 的归一化四元数。
 */
Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    
    double t_ab = tools::delta_time(t_a, t_b);
    double t_ac = tools::delta_time(t_a, t);
    
    double k = (t_ab > 1e-6) ? (t_ac / t_ab) : 0.0;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    
    // 如果目标时间过早，直接返回最老插值结果
    if (t < t_a) return q_c;
    // 找到包含目标时间的区间
    if (t > t_a && t <= t_b) return q_c;

    // 否则循环继续，搜索更靠后的帧
    continue;
  }
}

/**
 * @brief 发送视觉结算结果
 * 逻辑：封装协议包，计算 CRC16 校验码，发送至电控。
 * 虽然目前只发送 Yaw/Pitch，但结构体预留了速度项以供未来高阶追随算法。
 */
void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.pitch = VisionToGimbal.pitch;
  
  // 计算不含校验位自身的有效负载 CRC
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] 串口控制指令下放失败: {}", e.what());
  }
}

/**
 * @brief 接收线程实现
 * 逻辑：
 * 1. 采用字节轮推策略寻找 'S' 'P' 帧头。
 * 2. 补齐整包数据。
 * 3. 执行 CRC16 校验，若不符则丢弃。
 * 4. 更新共享状态，并将带时间戳的位姿压入插值队列。
 */
void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] 接收子线程启动。");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 5000) { // 连续读取超时，尝试软重连
      error_count = 0;
      tools::logger()->warn("[Gimbal] 通信中断，正在重新初始化串口...");
      reconnect();
      continue;
    }

    // 1. 读取帧头
    if (!read(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_.head))) {
      error_count++; continue;
    }
    if (rx_data_.head[0] != 'S' || rx_data_.head[1] != 'P') continue;

    auto t = std::chrono::steady_clock::now();

    // 2. 读取负载与校验位
    if (!read(reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
              sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++; continue;
    }

    // 3. 通信校验
    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
      tools::logger()->debug("[Gimbal] 数据包 CRC 校验失败（可能受干扰）。");
      continue;
    }

    error_count = 0;
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    queue_.push({q, t}); // 数据归档用于插值

    // 4. 共享状态同步
    std::lock_guard<std::mutex> lock(mutex_);
    state_.yaw = rx_data_.yaw;
    state_.pitch = rx_data_.pitch;
    state_.bullet_speed = rx_data_.bullet_speed;

    // 解析当前视觉任务模式 (由底盘操作手按键切换)
    switch (rx_data_.mode) {
      case 0: mode_ = GimbalMode::IDLE; break;
      case 1: mode_ = GimbalMode::AUTO_AIM; break;
      case 2: mode_ = GimbalMode::SMALL_BUFF; break;
      case 3: mode_ = GimbalMode::BIG_BUFF; break;
      default: mode_ = GimbalMode::IDLE; break;
    }
  }
}

/**
 * @brief 串口重连逻辑
 */
void Gimbal::reconnect()
{
  for (int i = 0; i < 10 && !quit_; ++i) {
    try {
      serial_.close();
      std::this_thread::sleep_for(1s);
      serial_.open();
      queue_.clear();
      tools::logger()->info("[Gimbal] 串口重连成功。");
      break;
    } catch (...) {
      tools::logger()->warn("[Gimbal] 正在尝试重连...");
    }
  }
}

}  // namespace io