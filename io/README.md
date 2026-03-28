# IO 模块文档

## 目录概述

`io/` 目录提供了整个视觉系统的硬件接口抽象层，包括相机、云台、CAN总线通信、IMU等硬件设备的封装。所有IO模块都位于 `io` 命名空间下，采用统一的接口设计，便于上层任务模块调用。

## 模块架构

```
io/
├── camera.*              # 相机抽象接口（工厂模式）
├── cboard.*              # CAN总线控制板通信
├── command.hpp           # 控制命令数据结构
├── gimbal/               # 云台串口通信
├── dm_imu/               # 达妙IMU串口通信
├── hikrobot/             # 海康威视工业相机实现
├── mindvision/           # 迈德威视工业相机实现
├── usbcamera/            # USB相机实现
├── socketcan.hpp          # SocketCAN实现
├── usb2can.hpp            # USB2CAN实现
├── serial/               # 串口通信库（第三方）
└── ros2/                 # ROS2通信接口（可选）
```

---

## 核心模块详解

### 1. Camera（相机抽象接口）

**文件**: `camera.hpp`, `camera.cpp`

**功能**: 
- 提供统一的相机接口，支持多种工业相机品牌
- 采用工厂模式，根据配置自动选择对应的相机实现
- 支持海康威视（HikRobot）和迈德威视（MindVision）相机

**接口**:
```cpp
class Camera {
public:
    Camera(const std::string & config_path);
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
};
```

**实现方式**:
- 内部维护 `CameraBase` 指针，根据配置文件的 `camera_name` 字段选择实现
- 支持的相机类型：
  - `"mindvision"`: 迈德威视相机（需要 `gamma`, `vid_pid` 参数）
  - `"hikrobot"`: 海康威视USB相机（需要 `gain`, `vid_pid` 参数）
  - `"hikrobot_gige"`: 海康威视GigE相机（需要 `gain` 参数）

**配置参数**:
- `camera_name`: 相机类型（必填）
- `exposure_ms`: 曝光时间（毫秒，必填）
- `gain`: 增益（海康威视相机）
- `gamma`: 伽马值（迈德威视相机）
- `vid_pid`: USB设备的VID:PID（USB相机）

**使用示例**:
```cpp
io::Camera camera(config_path);
cv::Mat img;
std::chrono::steady_clock::time_point timestamp;
camera.read(img, timestamp);
```

**被使用位置**:
- `src/standard.cpp` - 标准自瞄程序
- `src/standard_mpc.cpp` - MPC控制自瞄程序
- `src/sentry.cpp` - 哨兵程序
- `src/uav.cpp` - 无人机程序
- `tests/camera_test.cpp` - 相机测试程序
- 所有主程序和测试程序

---

### 2. CBoard（CAN总线控制板）

**文件**: `cboard.hpp`, `cboard.cpp`

**功能**:
- 通过CAN总线与控制板通信
- 接收IMU四元数数据（用于姿态解算）
- 接收弹速、模式、射击模式等信息
- 发送控制命令（yaw, pitch, 射击控制等）
- 支持SocketCAN和USB2CAN两种CAN接口

**接口**:
```cpp
class CBoard {
public:
    double bullet_speed;      // 弹速（m/s）
    Mode mode;                // 当前模式
    ShootMode shoot_mode;      // 射击模式（哨兵专用）
    double ft_angle;          // FT角度（无人机专用）
    
    CBoard(const std::string & config_path);
    Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
    void send(Command command) const;
};
```

**实现方式**:
- **CAN接口选择**: 优先尝试SocketCAN，失败后自动切换到USB2CAN
- **IMU数据队列**: 使用线程安全队列存储IMU四元数，支持时间戳插值查询
- **回调机制**: CAN接收采用回调函数，在独立线程中处理
- **自动重连**: SocketCAN和USB2CAN都实现了守护线程，自动检测连接状态并重连

**数据协议**:
- **接收**:
  - `quaternion_canid`: 接收IMU四元数（14位量化，范围[-1, 1]）
  - `bullet_speed_canid`: 接收弹速、模式、射击模式、FT角度
- **发送**:
  - `send_canid`: 发送控制命令（control, shoot, yaw, pitch, horizon_distance）

**配置参数**:
- `quaternion_canid`: IMU四元数CAN ID（必填）
- `bullet_speed_canid`: 弹速信息CAN ID（必填）
- `send_canid`: 控制命令CAN ID（必填）
- `can_interface`: SocketCAN接口名（如 "can0"）
- `usb2can_device`: USB2CAN设备路径（如 "/dev/ttyACM0"）
- `can_bitrate`: CAN波特率（默认1000000，支持500000/1000000/2000000/4000000）

**使用示例**:
```cpp
io::CBoard cboard(config_path);
auto q = cboard.imu_at(timestamp);  // 获取指定时间戳的IMU四元数
io::Command cmd;
cmd.control = true;
cmd.shoot = false;
cmd.yaw = 0.1;
cmd.pitch = -0.05;
cboard.send(cmd);  // 发送控制命令
```

**被使用位置**:
- `src/standard.cpp` - 标准自瞄程序
- `src/sentry.cpp` - 哨兵程序（多相机）
- `src/uav.cpp` - 无人机程序
- `src/mt_standard.cpp` - 多线程标准程序
- `tests/cboard_test.cpp` - CAN板测试程序
- 所有需要CAN通信的主程序

---

### 3. Gimbal（云台串口通信）

**文件**: `gimbal/gimbal.hpp`, `gimbal/gimbal.cpp`

**功能**:
- 通过串口与云台控制器通信
- 接收云台姿态、速度、弹速等信息
- 发送云台控制指令（yaw, pitch, 加速度等）
- 支持MPC控制模式（发送位置、速度、加速度）

**接口**:
```cpp
class Gimbal {
public:
    Gimbal(const std::string & config_path);
    ~Gimbal();
    
    GimbalMode mode() const;                    // 获取当前模式
    GimbalState state() const;                  // 获取云台状态
    Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);  // 获取指定时间的姿态四元数
    
    void send(bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, 
              float pitch, float pitch_vel, float pitch_acc);
    void send(io::VisionToGimbal VisionToGimbal);
};
```

**实现方式**:
- **串口通信**: 使用 `serial` 库进行串口读写
- **数据队列**: IMU四元数存储在队列中，支持时间戳插值
- **独立接收线程**: 后台线程持续接收云台数据
- **自动重连**: 检测到错误时自动重连串口

**数据协议**:
- **接收** (`GimbalToVision`):
  - 帧头: `'S'`, `'P'`
  - mode: 0=空闲, 1=自瞄, 2=小符, 3=大符
  - q[4]: 四元数（wxyz顺序）
  - yaw, yaw_vel, pitch, pitch_vel: 姿态和速度
  - bullet_speed: 弹速
  - bullet_count: 子弹计数
  - CRC16校验
- **发送** (`VisionToGimbal`):
  - 帧头: `'S'`, `'P'`
  - mode: 0=不控制, 1=控制不开火, 2=控制且开火
  - yaw, yaw_vel, yaw_acc: yaw轴位置/速度/加速度
  - pitch, pitch_vel, pitch_acc: pitch轴位置/速度/加速度
  - CRC16校验

**配置参数**:
- `com_port`: 串口设备路径（如 "/dev/gimbal"）

**使用示例**:
```cpp
io::Gimbal gimbal(config_path);
auto mode = gimbal.mode();  // 获取当前模式
auto state = gimbal.state();  // 获取云台状态
auto q = gimbal.q(timestamp);  // 获取指定时间的姿态

// 发送MPC控制指令
gimbal.send(true, false, yaw, yaw_vel, yaw_acc, pitch, pitch_vel, pitch_acc);
```

**被使用位置**:
- `src/standard_mpc.cpp` - MPC控制自瞄程序
- `src/auto_aim_debug_mpc.cpp` - MPC调试程序
- `src/auto_buff_debug_mpc.cpp` - BUFF MPC调试程序
- `tests/gimbal_test.cpp` - 云台测试程序
- `tests/fire_test.cpp` - 射击测试程序
- 所有使用MPC控制的程序

---

### 4. DM_IMU（达妙IMU）

**文件**: `dm_imu/dm_imu.hpp`, `dm_imu/dm_imu.cpp`

**功能**:
- 通过串口读取达妙IMU数据
- 解析加速度、角速度、欧拉角
- 将欧拉角转换为四元数
- 支持时间戳插值查询

**接口**:
```cpp
class DM_IMU {
public:
    DM_IMU();
    ~DM_IMU();
    Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
};
```

**实现方式**:
- **串口通信**: 固定使用 `/dev/ttyACM0`，波特率921600
- **数据解析**: 解析三帧数据（加速度、角速度、欧拉角），每帧16字节
- **CRC校验**: 每帧数据都有CRC16校验
- **四元数转换**: 将欧拉角（roll, pitch, yaw）转换为四元数
- **独立接收线程**: 后台线程持续读取IMU数据

**数据协议**:
- 帧结构包含三个子帧：
  1. 加速度帧（accx, accy, accz）
  2. 角速度帧（gyrox, gyroy, gyroz）
  3. 欧拉角帧（roll, pitch, yaw，单位：度）

**配置参数**:
- 无（硬编码串口路径和波特率）

**使用示例**:
```cpp
io::DM_IMU imu;
auto q = imu.imu_at(timestamp);  // 获取指定时间的IMU四元数
```

**被使用位置**:
- `tests/dm_test.cpp` - IMU测试程序
- 需要独立IMU数据的场景

---
### 5. USBCamera（USB相机）

**文件**: `usbcamera/usbcamera.hpp`, `usbcamera/usbcamera.cpp`

**功能**:
- 封装OpenCV VideoCapture用于USB相机
- 支持多相机同时使用（通过设备名区分，如 "video0", "video2"）
- 自动重连机制
- 可配置曝光、帧率、分辨率等参数

**接口**:
```cpp
class USBCamera {
public:
    USBCamera(const std::string & open_name, const std::string & config_path);
    ~USBCamera();
    cv::Mat read();
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp);
    std::string device_name;  // 设备名称
};
```

**实现方式**:
- **OpenCV封装**: 使用 `cv::VideoCapture` 打开USB相机
  - 设备名格式: "video0", "video1", "video2" 等
  - 内部转换为 `/dev/video0`, `/dev/video1` 等设备路径
- **独立采集线程**: 后台线程持续采集图像到队列
  - 使用互斥锁保护 `cv::VideoCapture` 对象
  - 采集的图像和时间戳存储在线程安全队列中
- **守护线程**: 检测连接状态，自动重连
  - 每100ms检查一次连接状态
  - 如果采集失败，尝试重新打开相机
  - 最多重试20次，超过后放弃
- **参数配置**: 支持曝光、帧率、分辨率、增益、伽马等参数
  - 通过OpenCV的 `cap.set()` 接口设置参数

**配置参数**:
- `image_width`, `image_height`: 图像分辨率（必填）
- `fov_h`, `fov_v`: 水平/垂直视场角（可选）
- `usb_frame_rate`: 帧率（必填）
- `usb_exposure`: 曝光值（必填，范围1-80000）
- `usb_gamma`: 伽马值（必填）
- `usb_gain`: 增益（必填，范围0-96）

**使用示例**:
```cpp
io::USBCamera usbcam1("video0", config_path);
io::USBCamera usbcam2("video2", config_path);
cv::Mat img;
std::chrono::steady_clock::time_point timestamp;
usbcam1.read(img, timestamp);
```

**注意事项**:
- 设备名必须是 "video" 开头的字符串
- 多个相机需要不同的设备名
- 如果相机打开失败，会持续重试直到成功或达到最大重试次数

**被使用位置**:
- `src/sentry.cpp` - 哨兵程序（多相机）
- `tests/multi_usbcamera_test.cpp` - 多相机测试程序

---

### 6. SocketCAN（SocketCAN实现）

**文件**: `socketcan.hpp`

**功能**:
- Linux SocketCAN接口封装
- 支持CAN帧的发送和接收
- 自动重连机制
- 使用epoll实现高效的事件驱动接收

**接口**:
```cpp
class SocketCAN {
public:
    SocketCAN(const std::string & interface, 
              std::function<void(const can_frame & frame)> rx_handler);
    ~SocketCAN();
    void write(can_frame * frame) const;
};
```

**实现方式**:
- **Socket接口**: 使用 `PF_CAN` 套接字
- **epoll机制**: 使用epoll实现非阻塞接收
- **独立接收线程**: 后台线程处理CAN帧接收
- **守护线程**: 检测连接状态，自动重连

**使用场景**:
- 被 `CBoard` 内部使用
- 需要直接使用SocketCAN的场景

---

### 7. USB2CAN（USB2CAN实现）

**文件**: `usb2can.hpp`

**功能**:
- 达妙科技USB2CAN设备封装
- 支持FDCAN协议
- 可配置波特率（500K/1M/2M/4M）
- 自动重连机制

**接口**:
```cpp
class USB2CAN {
public:
    USB2CAN(const std::string & device_name, 
            std::function<void(const can_frame & frame)> rx_handler,
            uint32_t bitrate);
    ~USB2CAN();
    void write(can_frame * frame) const;
    bool is_ok() const;
};
```

**实现方式**:
- **USB设备**: 使用达妙科技USB2FDCAN SDK
- **独立接收线程**: 后台线程持续读取CAN帧
- **守护线程**: 检测连接状态，自动重连
- **波特率配置**: 支持500K/1M/2M/4M四种波特率

**依赖**:
- 需要链接 `libusb_fdcan.so`（位于 `/home/rm/DingLab/USB2FDCAN_SDK/lib/`）

**使用场景**:
- 被 `CBoard` 内部使用
- 当系统没有SocketCAN接口时使用

---

### 8. Command（控制命令）

**文件**: `command.hpp`

**功能**:
- 定义发送给控制板的标准命令结构
- 包含控制标志、射击标志、姿态指令等

**数据结构**:
```cpp
struct Command {
    bool control;              // 是否控制云台
    bool shoot;               // 是否射击
    double yaw;               // yaw角度（弧度）
    double pitch;             // pitch角度（弧度）
    double horizon_distance;  // 水平距离（米，无人机专用）
};
```

**使用场景**:
- `CBoard::send()` 的参数
- 所有需要发送控制命令的地方

---

### 9. ROS2（ROS2通信接口）

**文件**: `ros2/ros2.hpp`, `ros2/ros2.cpp`, `ros2/publish2nav.hpp`, `ros2/publish2nav.cpp`, `ros2/subscribe2nav.hpp`, `ros2/subscribe2nav.cpp`

**功能**:
- 提供ROS2通信接口，用于与导航模块通信
- 发布目标位置到导航模块
- 订阅敌方状态和自瞄目标信息

**接口**:
```cpp
class ROS2 {
public:
    ROS2();
    ~ROS2();
    void publish(const Eigen::Vector4d & target_pos);
    std::vector<int8_t> subscribe_enemy_status();
    std::vector<int8_t> subscribe_autoaim_target();
};
```

**实现方式**:
- **条件编译**: 仅在检测到ROS2环境时编译
- **独立线程**: 使用两个独立线程分别spin发布和订阅节点
- **消息类型**: 使用自定义消息类型 `sp_msgs`
- **节点管理**: 内部管理 `Publish2Nav` 和 `Subscribe2Nav` 两个ROS节点

**子模块**:

#### 9.1 Publish2Nav（发布到导航模块）

**文件**: `ros2/publish2nav.hpp`, `ros2/publish2nav.cpp`

**功能**:
- 发布目标位置信息到导航模块
- 节点名: `auto_aim_target_pos_publisher`
- Topic: `auto_aim_target_pos` (std_msgs::msg::String)

**接口**:
```cpp
class Publish2Nav : public rclcpp::Node {
public:
    Publish2Nav();
    ~Publish2Nav();
    void start();
    void send_data(const Eigen::Vector4d & data);
};
```

**数据格式**:
- 将 `Eigen::Vector4d` 转换为逗号分隔的字符串: `"x,y,z,w"`

**使用方式**:
- 由 `ROS2` 类内部管理，通过 `ROS2::publish()` 调用

#### 9.2 Subscribe2Nav（从导航模块订阅）

**文件**: `ros2/subscribe2nav.hpp`, `ros2/subscribe2nav.cpp`

**功能**:
- 订阅敌方状态信息
- 订阅自瞄目标信息
- 节点名: `nav_subscriber`

**接口**:
```cpp
class Subscribe2Nav : public rclcpp::Node {
public:
    Subscribe2Nav();
    ~Subscribe2Nav();
    void start();
    std::vector<int8_t> subscribe_enemy_status();
    std::vector<int8_t> subscribe_autoaim_target();
};
```

**订阅的Topic**:
- `enemy_status` (sp_msgs::msg::EnemyStatusMsg): 敌方状态，包含无敌状态敌方ID列表
- `autoaim_target` (sp_msgs::msg::AutoaimTargetMsg): 自瞄目标，包含目标ID列表

**实现特性**:
- **线程安全队列**: 使用 `ThreadSafeQueue` 存储接收到的消息
- **超时清理**: 如果超过1.5秒未收到新消息，自动清空队列
- **最新消息**: 始终返回队列中的最新消息

**使用方式**:
- 由 `ROS2` 类内部管理，通过 `ROS2::subscribe_enemy_status()` 和 `ROS2::subscribe_autoaim_target()` 调用

**依赖**:
- ROS2 (rclcpp, std_msgs, sp_msgs)
- 仅在ROS2环境中可用

**使用场景**:
- `src/sentry.cpp` - 哨兵程序（与导航模块通信）

---

## 相机实现模块

### HikRobot（海康威视相机）

**文件**: `hikrobot/hikrobot.hpp`, `hikrobot/hikrobot.cpp`

**功能**:
- 海康威视工业相机实现
- 支持USB和GigE两种接口
- 使用海康威视MVS SDK

**接口**:
```cpp
class HikRobot : public CameraBase {
public:
    HikRobot(double exposure_ms, double gain, const std::string & vid_pid);  // USB相机
    HikRobot(double exposure_ms, double gain);  // GigE相机
    ~HikRobot();
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
};
```

**实现方式**:
- **USB相机模式**:
  - 需要指定VID:PID（格式: "vid:pid"，如 "2bdf:0001"）
  - 使用libusb进行USB设备枚举和选择
  - 支持USB设备重置和重连
- **GigE相机模式**:
  - 自动发现网络上的GigE相机
  - 无需指定VID:PID
- **采集机制**:
  - 独立采集线程持续从相机获取图像
  - 使用线程安全队列存储图像数据
  - 守护线程监控采集状态，失败时自动重连
- **参数配置**:
  - 曝光时间（微秒）
  - 增益值
  - 自动设置触发模式、图像格式等

**特性**:
- USB相机：需要指定VID:PID
- GigE相机：自动发现网络相机
- 独立采集线程
- 守护线程自动重连
- 支持USB设备重置

**依赖**:
- `libMvCameraControl.so`（海康威视SDK，位于 `hikrobot/lib/`）
- `libusb-1.0`（USB设备操作）

**使用场景**:
- 通过 `io::Camera` 工厂类使用
- 配置 `camera_name: "hikrobot"` 或 `"hikrobot_gige"`

---

### MindVision（迈德威视相机）

**文件**: `mindvision/mindvision.hpp`, `mindvision/mindvision.cpp`

**功能**:
- 迈德威视工业相机实现
- 使用迈德威视MVSDK

**接口**:
```cpp
class MindVision : public CameraBase {
public:
    MindVision(double exposure_ms, double gamma, const std::string & vid_pid);
    ~MindVision();
    void read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp) override;
};
```

**实现方式**:
- **设备枚举**: 使用MVSDK枚举所有连接的相机
- **VID:PID匹配**: 根据配置的VID:PID选择对应相机
- **参数配置**:
  - 曝光时间（毫秒）
  - 伽马值
  - 自动关闭自动曝光
  - 设置图像格式和分辨率
- **采集机制**:
  - 独立采集线程持续从相机获取图像
  - 使用线程安全队列存储图像数据
  - 守护线程监控采集状态，失败时自动重连
- **USB重置**: 支持USB设备重置以恢复连接

**特性**:
- 需要指定VID:PID
- 支持曝光和伽马参数配置
- 独立采集线程
- 守护线程自动重连
- 支持USB设备重置

**依赖**:
- `libMVSDK.so`（迈德威视SDK，位于 `mindvision/lib/`）
- `libusb-1.0`（USB设备操作）

**使用场景**:
- 通过 `io::Camera` 工厂类使用
- 配置 `camera_name: "mindvision"`

---

### Serial（串口通信库）

**文件**: `serial/` 目录（第三方库）

**功能**:
- 跨平台串口通信库
- 支持Linux、Windows、macOS
- 提供统一的串口操作接口

**目录结构**:
```
serial/
├── CMakeLists.txt
├── include/serial/
│   ├── serial.h          # 主头文件
│   ├── v8stdint.h        # 标准整数类型定义
│   └── impl/
│       ├── unix.h        # Unix/Linux实现
│       └── win.h         # Windows实现
└── src/
    ├── serial.cc          # 主实现文件
    └── impl/
        ├── unix.cc       # Unix/Linux实现
        ├── win.cc        # Windows实现
        └── list_ports/   # 串口列表功能
```

**主要接口**:
```cpp
class Serial {
public:
    Serial();
    Serial(const std::string &port, uint32_t baudrate, ...);
    void setPort(const std::string &port);
    void setBaudrate(uint32_t baudrate);
    void open();
    void close();
    size_t read(uint8_t *buffer, size_t size);
    size_t write(const uint8_t *data, size_t size);
    bool isOpen() const;
};
```

**使用场景**:
- 被 `io::Gimbal` 使用（云台串口通信）
- 被 `io::DM_IMU` 使用（IMU串口通信）
- 所有需要串口通信的模块

**特性**:
- 跨平台支持
- 线程安全（需要外部同步）
- 支持超时设置
- 支持流控制、校验位、停止位等配置

---

## 使用模式总结

### 标准自瞄流程
```cpp
// 1. 初始化硬件
io::CBoard cboard(config_path);
io::Camera camera(config_path);

// 2. 读取数据
cv::Mat img;
std::chrono::steady_clock::time_point t;
camera.read(img, t);
auto q = cboard.imu_at(t);

// 3. 处理数据（检测、跟踪、解算）
// ...

// 4. 发送控制命令
io::Command cmd;
cmd.control = true;
cmd.shoot = false;
cmd.yaw = yaw_angle;
cmd.pitch = pitch_angle;
cboard.send(cmd);
```

### MPC控制流程
```cpp
// 1. 初始化硬件
io::Gimbal gimbal(config_path);
io::Camera camera(config_path);

// 2. 读取数据
cv::Mat img;
std::chrono::steady_clock::time_point t;
camera.read(img, t);
auto q = gimbal.q(t);
auto state = gimbal.state();

// 3. MPC规划
// ...

// 4. 发送MPC指令
gimbal.send(true, false, yaw, yaw_vel, yaw_acc, 
            pitch, pitch_vel, pitch_acc);
```

---

## 配置参数参考

### 相机配置
```yaml
camera_name: "hikrobot_gige"  # 或 "hikrobot", "mindvision"
exposure_ms: 0.8
gain: 16.9
# vid_pid: "2bdf:0001"  # USB相机需要
```

### CAN配置
```yaml
quaternion_canid: 0x01
bullet_speed_canid: 0x110
send_canid: 0xff
can_interface: "can0"  # SocketCAN
# usb2can_device: "/dev/ttyACM0"  # USB2CAN
# can_bitrate: 1000000  # USB2CAN波特率
```

### 云台配置
```yaml
com_port: "/dev/gimbal"
```

---

## 依赖关系

```
io (静态库)
├── yaml-cpp (配置解析)
├── serial (串口库)
├── MvCameraControl (海康威视SDK)
├── MVSDK (迈德威视SDK)
├── usb-1.0 (USB库)
└── usb_fdcan (达妙USB2CAN SDK)
```

---

## 注意事项

1. **CAN接口优先级**: CBoard优先使用SocketCAN，失败后自动切换到USB2CAN
2. **线程安全**: 所有IO模块都使用线程安全的数据结构，支持多线程访问
3. **自动重连**: SocketCAN、USB2CAN、相机等都有自动重连机制
4. **时间戳插值**: IMU和云台数据支持时间戳插值，确保数据同步
5. **ROS2可选**: ROS2模块仅在检测到ROS2环境时编译
6. **配置验证**: 所有模块在构造时都会验证配置参数，缺少必要参数会抛出异常

---

## 测试程序

- `tests/camera_test.cpp` - 相机测试
- `tests/cboard_test.cpp` - CAN板测试
- `tests/gimbal_test.cpp` - 云台测试
- `tests/dm_test.cpp` - IMU测试
- `tests/multi_usbcamera_test.cpp` - 多相机测试

---

## 更新日志

- 支持USB2CAN自动检测和重连
- 支持CAN波特率配置
- 云台支持MPC控制模式
- 相机支持GigE接口
- ROS2模块支持条件编译

