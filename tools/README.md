# Tools 工具库文档

> **作者**: 项目创始人  
> **更新日期**: 2025年  
> **目标读者**: 刚入门的学弟学妹

## 📚 目录

- [概述](#概述)
- [工具分类](#工具分类)
- [详细文档](#详细文档)
  - [1. 日志系统 (logger)](#1-日志系统-logger)
  - [2. 优雅退出 (exiter)](#2-优雅退出-exiter)
  - [3. 线程安全队列 (thread_safe_queue)](#3-线程安全队列-thread_safe_queue)
  - [4. 数学工具 (math_tools)](#4-数学工具-math_tools)
  - [5. 配置文件读取 (yaml)](#5-配置文件读取-yaml)
  - [6. 弹道计算 (trajectory)](#6-弹道计算-trajectory)
  - [7. 扩展卡尔曼滤波 (extended_kalman_filter)](#7-扩展卡尔曼滤波-extended_kalman_filter)
  - [8. PID控制器 (pid)](#8-pid控制器-pid)
  - [9. 数据记录 (recorder)](#9-数据记录-recorder)
  - [10. 可视化绘图 (plotter)](#10-可视化绘图-plotter)
  - [11. 图像工具 (img_tools)](#11-图像工具-img_tools)
  - [12. CRC校验 (crc)](#12-crc校验-crc)
  - [13. RANSAC正弦拟合 (ransac_sine_fitter)](#13-ransac正弦拟合-ransac_sine_fitter)
  - [14. 线程池 (thread_pool)](#14-线程池-thread_pool)
- [使用建议](#使用建议)
- [常见问题](#常见问题)

---

## 概述

`tools/` 目录是项目的**工具库**，包含了整个视觉系统所需的基础工具和通用功能。这些工具被项目的各个模块广泛使用，是系统的基础设施。

### 设计理念

- **单一职责**: 每个工具只负责一个明确的功能
- **线程安全**: 多线程环境下可以安全使用
- **易于使用**: 提供简洁的API接口
- **高性能**: 针对实时系统优化

---

## 工具分类

| 类别 | 工具名称 | 主要用途 |
|------|---------|---------|
| **基础工具** | logger, exiter, yaml | 日志、退出控制、配置读取 |
| **并发工具** | thread_safe_queue, thread_pool | 线程间通信、任务调度 |
| **数学工具** | math_tools, trajectory, extended_kalman_filter, pid | 坐标转换、弹道计算、滤波、控制 |
| **调试工具** | recorder, plotter, img_tools | 数据记录、可视化、图像绘制 |
| **通信工具** | crc | 数据校验 |
| **算法工具** | ransac_sine_fitter | 正弦拟合 |

---

## 详细文档

### 1. 日志系统 (logger)

**文件**: `logger.hpp`, `logger.cpp`

**功能**: 提供统一的日志记录功能，支持同时输出到控制台和文件。

**特点**:
- 自动创建带时间戳的日志文件（保存在 `logs/` 目录）
- 支持多级别日志（debug, info, warn, error）
- 线程安全，可在多线程环境下使用
- 使用 `spdlog` 库实现

**使用示例**:
```cpp
#include "tools/logger.hpp"

// 记录不同级别的日志
tools::logger()->debug("调试信息: {}", value);
tools::logger()->info("普通信息: {}", message);
tools::logger()->warn("警告信息: {}", warning);
tools::logger()->error("错误信息: {}", error);
```

**在项目中的使用**:
- ✅ **所有模块**: 几乎每个 `.cpp` 文件都使用了 logger
- 📍 **主要位置**: 
  - `io/cboard.cpp` - CAN通信日志
  - `tasks/auto_aim/tracker.cpp` - 跟踪器日志
  - `src/standard.cpp` - 主程序日志

**为什么需要它**:
- 调试时可以看到程序运行状态
- 记录错误信息便于排查问题
- 日志文件可以用于后续分析

---

### 2. 优雅退出 (exiter)

**文件**: `exiter.hpp`, `exiter.cpp`

**功能**: 捕获 Ctrl+C 信号，实现程序的优雅退出。

**特点**:
- 监听 SIGINT 信号（Ctrl+C）
- 全局单例模式，确保只有一个实例
- 线程安全

**使用示例**:
```cpp
#include "tools/exiter.hpp"

tools::Exiter exiter;

while (!exiter.exit()) {
    // 主循环
    // 当用户按 Ctrl+C 时，exiter.exit() 会返回 true
    do_work();
}
```

**在项目中的使用**:
- ✅ **所有主程序**: `src/` 目录下的所有主程序
- 📍 **典型位置**:
  - `src/standard.cpp` - 标准自瞄主程序
  - `src/sentry.cpp` - 哨兵主程序
  - `src/uav.cpp` - 无人机主程序

**为什么需要它**:
- 避免强制终止导致的数据丢失
- 可以安全地清理资源（关闭文件、释放内存等）
- 提供良好的用户体验

---

### 3. 线程安全队列 (thread_safe_queue)

**文件**: `thread_safe_queue.hpp`

**功能**: 提供线程安全的队列，用于多线程之间的数据传递。

**特点**:
- 使用互斥锁和条件变量保证线程安全
- 支持阻塞式 `pop()`（队列为空时等待）
- 可配置最大容量
- 支持队列满时的处理策略（丢弃旧数据或拒绝新数据）

**使用示例**:
```cpp
#include "tools/thread_safe_queue.hpp"

// 创建队列，最大容量1000
tools::ThreadSafeQueue<MyData> queue(1000);

// 生产者线程
void producer() {
    MyData data;
    queue.push(data);  // 线程安全地推入数据
}

// 消费者线程
void consumer() {
    MyData data;
    queue.pop(data);  // 线程安全地取出数据（如果队列为空会阻塞等待）
}
```

**在项目中的使用**:
- ✅ **IO模块**: 相机、IMU、云台等数据采集
- ✅ **任务模块**: 多线程检测、跟踪等
- 📍 **典型位置**:
  - `io/cboard.cpp` - IMU数据队列
  - `io/hikrobot/hikrobot.cpp` - 相机图像队列
  - `io/gimbal/gimbal.cpp` - 云台姿态队列
  - `tasks/auto_aim/multithread/mt_detector.cpp` - 多线程检测队列

**为什么需要它**:
- 相机采集线程和主处理线程需要安全地传递图像数据
- IMU数据采集线程和主线程需要传递姿态数据
- 避免数据竞争和内存安全问题

---

### 4. 数学工具 (math_tools)

**文件**: `math_tools.hpp`, `math_tools.cpp`

**功能**: 提供常用的数学计算函数，包括坐标转换、角度处理等。

**主要函数**:

#### 4.1 角度限制 `limit_rad()`
```cpp
double limit_rad(double angle);
```
- **功能**: 将角度限制在 (-π, π] 范围内
- **使用**: 处理角度归一化，避免角度溢出

#### 4.2 四元数/旋转矩阵转欧拉角 `eulers()`
```cpp
Eigen::Vector3d eulers(Eigen::Quaterniond q, int axis0, int axis1, int axis2);
Eigen::Vector3d eulers(Eigen::Matrix3d R, int axis0, int axis1, int axis2);
```
- **功能**: 将四元数或旋转矩阵转换为欧拉角（yaw, pitch, roll）
- **参数**: `axis0, axis1, axis2` 指定旋转顺序（0=x, 1=y, 2=z）
- **使用**: 云台姿态表示、坐标变换

#### 4.3 欧拉角转旋转矩阵 `rotation_matrix()`
```cpp
Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d & ypr);
```
- **功能**: 将欧拉角转换为旋转矩阵
- **使用**: 坐标变换计算

#### 4.4 坐标转换 `xyz2ypd()` / `ypd2xyz()`
```cpp
Eigen::Vector3d xyz2ypd(const Eigen::Vector3d & xyz);  // 直角坐标转球坐标
Eigen::Vector3d ypd2xyz(const Eigen::Vector3d & ypd);  // 球坐标转直角坐标
```
- **功能**: 在直角坐标系和球坐标系之间转换
- **ypd**: yaw（偏航角）、pitch（俯仰角）、distance（距离）
- **使用**: 目标位置表示、弹道计算

#### 4.5 时间差计算 `delta_time()`
```cpp
double delta_time(
    const std::chrono::steady_clock::time_point & a,
    const std::chrono::steady_clock::time_point & b
);
```
- **功能**: 计算两个时间点之间的差值（秒）
- **使用**: 性能测量、时间同步、运动预测

**使用示例**:
```cpp
#include "tools/math_tools.hpp"

// 角度归一化
double angle = 3.5;  // 超过π
angle = tools::limit_rad(angle);  // 归一化到 (-π, π]

// 四元数转欧拉角（ZYX顺序）
Eigen::Quaterniond q(1, 0, 0, 0);
Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0);  // yaw, pitch, roll

// 坐标转换
Eigen::Vector3d xyz(1.0, 2.0, 3.0);
Eigen::Vector3d ypd = tools::xyz2ypd(xyz);  // 转换为球坐标

// 时间差计算
auto t1 = std::chrono::steady_clock::now();
// ... 做一些操作
auto t2 = std::chrono::steady_clock::now();
double dt = tools::delta_time(t2, t1);  // 计算耗时
```

**在项目中的使用**:
- ✅ **几乎所有模块**: 数学工具是项目的基础
- 📍 **典型位置**:
  - `tasks/auto_aim/solver.cpp` - 坐标变换、重投影
  - `tasks/auto_aim/tracker.cpp` - 时间差计算、角度处理
  - `tasks/auto_aim/target.cpp` - 坐标转换、预测
  - `io/gimbal/gimbal.cpp` - 姿态转换

**为什么需要它**:
- 机器人视觉系统需要大量的坐标变换（相机坐标系、世界坐标系、云台坐标系）
- 角度计算需要规范化处理
- 时间同步和性能测量是实时系统的关键

---

### 5. 配置文件读取 (yaml)

**文件**: `yaml.hpp`

**功能**: 提供简洁的YAML配置文件读取接口。

**特点**:
- 封装了 `yaml-cpp` 库
- 自动错误处理和日志记录
- 类型安全的读取接口

**使用示例**:
```cpp
#include "tools/yaml.hpp"

// 加载配置文件
auto yaml = tools::load("configs/standard.yaml");

// 读取配置项（自动类型转换）
std::string camera_name = tools::read<std::string>(yaml, "camera_name");
int width = tools::read<int>(yaml, "width");
double exposure = tools::read<double>(yaml, "exposure_ms");
```

**在项目中的使用**:
- ✅ **所有模块**: 配置文件的读取
- 📍 **典型位置**:
  - `io/camera.cpp` - 读取相机配置
  - `io/cboard.cpp` - 读取CAN配置
  - `tasks/auto_aim/yolo.cpp` - 读取YOLO模型配置
  - `tasks/auto_aim/solver.cpp` - 读取相机标定参数

**为什么需要它**:
- 项目使用YAML格式存储配置（相机参数、模型路径、算法参数等）
- 统一的配置读取接口，便于维护
- 自动错误处理，避免配置错误导致程序崩溃

---

### 6. 弹道计算 (trajectory)

**文件**: `trajectory.hpp`, `trajectory.cpp`

**功能**: 计算弹丸的飞行轨迹，不考虑空气阻力。

**特点**:
- 基于物理学的抛体运动公式
- 计算飞行时间和发射角度
- 自动选择飞行时间最短的解

**使用示例**:
```cpp
#include "tools/trajectory.hpp"

// 计算弹道
// v0: 子弹初速度 (m/s)
// d: 目标水平距离 (m)
// h: 目标竖直高度 (m)
double bullet_speed = 25.0;  // 25 m/s
double distance = 5.0;        // 5米
double height = 0.5;          // 0.5米高

tools::Trajectory traj(bullet_speed, distance, height);

if (!traj.unsolvable) {
    double pitch = traj.pitch;      // 需要的发射角度（弧度）
    double fly_time = traj.fly_time; // 飞行时间（秒）
} else {
    // 无解（目标太远或太高）
}
```

**在项目中的使用**:
- ✅ **瞄准模块**: 计算瞄准角度
- 📍 **典型位置**:
  - `tasks/auto_aim/aimer.cpp` - 计算瞄准点
  - `tasks/auto_buff/buff_aimer.cpp` - BUFF目标瞄准

**为什么需要它**:
- 弹丸受重力影响会下坠，需要提前计算补偿角度
- 确保弹丸能够命中目标
- 是自瞄系统的核心算法之一

---

### 7. 扩展卡尔曼滤波 (extended_kalman_filter)

**文件**: `extended_kalman_filter.hpp`, `extended_kalman_filter.cpp`

**功能**: 实现扩展卡尔曼滤波器（EKF），用于目标状态估计和预测。

**特点**:
- 支持非线性系统
- 提供预测（predict）和更新（update）接口
- 包含卡方检验（NIS/NEES）用于异常检测
- 支持自定义状态加法函数（用于角度等周期性状态）

**使用示例**:
```cpp
#include "tools/extended_kalman_filter.hpp"

// 初始化滤波器
Eigen::VectorXd x0(11);  // 状态向量 [x, vx, y, vy, z, vz, yaw, w, r, l, h]
x0 << 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1;
Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(11, 11) * 0.1;  // 初始协方差

tools::ExtendedKalmanFilter ekf(x0, P0);

// 预测步骤
Eigen::MatrixXd F = ...;  // 状态转移矩阵
Eigen::MatrixXd Q = ...;  // 过程噪声协方差
ekf.predict(F, Q);

// 更新步骤
Eigen::VectorXd z = ...;  // 观测值
Eigen::MatrixXd H = ...;  // 观测矩阵
Eigen::MatrixXd R = ...;  // 观测噪声协方差
ekf.update(z, H, R);

// 获取状态估计
Eigen::VectorXd x = ekf.x;
```

**在项目中的使用**:
- ✅ **目标跟踪**: 用于跟踪目标的位置、速度、姿态
- 📍 **典型位置**:
  - `tasks/auto_aim/target.cpp` - 目标状态估计
  - `tasks/auto_buff/buff_target.cpp` - BUFF目标预测

**为什么需要它**:
- 目标检测存在噪声和抖动
- 需要平滑的目标轨迹用于预测
- 可以预测目标的未来位置，提高命中率

---

### 8. PID控制器 (pid)

**文件**: `pid.hpp`, `pid.cpp`

**功能**: 实现PID控制器，用于云台控制。

**特点**:
- 标准PID算法（比例、积分、微分）
- 支持角度模式（自动处理角度周期性）
- 可配置输出限制和积分限制

**使用示例**:
```cpp
#include "tools/pid.hpp"

// 创建PID控制器
// dt: 控制周期 (s)
// kp, ki, kd: PID参数
// max_out: 最大输出
// max_iout: 积分项最大输出
// angular: 是否为角度控制（true=角度模式）
tools::PID pid(0.01, 1.0, 0.1, 0.05, 10.0, 5.0, true);

// 计算控制输出
float target = 1.0;   // 目标值
float feedback = 0.5; // 反馈值
float output = pid.calc(target, feedback);  // PID输出

// 调试信息
float p_out = pid.pout;  // P项输出
float i_out = pid.iout;  // I项输出
float d_out = pid.dout;  // D项输出
```

**在项目中的使用**:
- ✅ **控制模块**: 云台控制（虽然当前项目主要通过CAN直接发送角度）
- 📍 **潜在位置**: 可用于云台速度控制、位置控制

**为什么需要它**:
- PID是经典的控制算法
- 可以用于云台的平滑控制
- 虽然当前项目主要使用位置控制，但PID可用于速度控制模式

---

### 9. 数据记录 (recorder)

**文件**: `recorder.hpp`, `recorder.cpp`

**功能**: 记录图像、姿态和时间戳，用于后续回放和分析。

**特点**:
- 异步记录（不阻塞主线程）
- 同时保存视频文件和文本文件（姿态数据）
- 可配置帧率
- 使用线程安全队列缓冲数据

**使用示例**:
```cpp
#include "tools/recorder.hpp"

// 创建记录器（30 FPS）
tools::Recorder recorder(30);

// 记录数据
cv::Mat img;
Eigen::Quaterniond q;
std::chrono::steady_clock::time_point timestamp;

recorder.record(img, q, timestamp);
```

**在项目中的使用**:
- ✅ **主程序**: 用于记录测试数据
- 📍 **典型位置**:
  - `src/standard_mpc.cpp` - 记录MPC测试数据
  - `src/sentry_multithread.cpp` - 记录哨兵数据

**为什么需要它**:
- 记录测试数据用于后续分析
- 可以回放数据，方便调试算法
- 保存姿态数据用于离线分析

---

### 10. 可视化绘图 (plotter)

**文件**: `plotter.hpp`, `plotter.cpp`

**功能**: 通过UDP发送JSON数据到可视化工具（如PlotJuggler）进行实时绘图。

**特点**:
- 使用UDP协议发送数据
- 数据格式为JSON
- 线程安全
- 用于实时可视化调试数据

**使用示例**:
```cpp
#include "tools/plotter.hpp"
#include <nlohmann/json.hpp>

// 创建绘图器（默认发送到 127.0.0.1:9870）
tools::Plotter plotter;

// 准备数据
nlohmann::json data;
data["x"] = 1.0;
data["y"] = 2.0;
data["velocity"] = 3.0;

// 发送数据
plotter.plot(data);
```

**在项目中的使用**:
- ✅ **调试程序**: 用于可视化调试数据
- 📍 **典型位置**:
  - `src/auto_aim_debug_mpc.cpp` - 可视化MPC调试数据
  - `src/sentry_debug.cpp` - 可视化哨兵调试数据
  - `tests/auto_aim_test.cpp` - 测试数据可视化

**为什么需要它**:
- 实时查看算法运行状态
- 可视化目标位置、速度、角度等数据
- 便于调试和优化算法

---

### 11. 图像工具 (img_tools)

**文件**: `img_tools.hpp`, `img_tools.cpp`

**功能**: 提供图像绘制工具，用于在图像上绘制点、线和文本。

**主要函数**:

#### 11.1 绘制点 `draw_point()`
```cpp
void draw_point(cv::Mat & img, const cv::Point & point, 
                const cv::Scalar & color = {0, 0, 255}, int radius = 3);
```

#### 11.2 绘制多个点 `draw_points()`
```cpp
void draw_points(cv::Mat & img, const std::vector<cv::Point> & points,
                 const cv::Scalar & color = {0, 0, 255}, int thickness = 2);
```

#### 11.3 绘制文本 `draw_text()`
```cpp
void draw_text(cv::Mat & img, const std::string & text, const cv::Point & point,
               const cv::Scalar & color = {0, 255, 255}, 
               double font_scale = 1.0, int thickness = 2);
```

**使用示例**:
```cpp
#include "tools/img_tools.hpp"

cv::Mat img;

// 绘制单个点
tools::draw_point(img, cv::Point(100, 100), {0, 255, 0}, 5);

// 绘制多个点（如装甲板角点）
std::vector<cv::Point> corners = {...};
tools::draw_points(img, corners, {255, 0, 0}, 2);

// 绘制文本
tools::draw_text(img, "Target Found", cv::Point(10, 30), {0, 255, 255});
```

**在项目中的使用**:
- ✅ **调试和可视化**: 在图像上绘制检测结果
- 📍 **典型位置**:
  - `tasks/auto_aim/solver.cpp` - 绘制重投影点
  - `tests/auto_aim_test.cpp` - 绘制检测结果
  - `src/sentry_debug.cpp` - 调试可视化

**为什么需要它**:
- 可视化检测结果，便于调试
- 在图像上标注目标位置、角度等信息
- 提供统一的绘制接口，代码更简洁

---

### 12. CRC校验 (crc)

**文件**: `crc.hpp`, `crc.cpp`

**功能**: 提供CRC8和CRC16校验功能，用于数据完整性检查。

**主要函数**:
```cpp
uint8_t get_crc8(const uint8_t * data, uint16_t len);   // 计算CRC8
bool check_crc8(const uint8_t * data, uint16_t len);    // 校验CRC8

uint16_t get_crc16(const uint8_t * data, uint32_t len); // 计算CRC16
bool check_crc16(const uint8_t * data, uint32_t len);   // 校验CRC16
```

**使用示例**:
```cpp
#include "tools/crc.hpp"

uint8_t data[10] = {...};

// 计算CRC16
uint16_t crc = tools::get_crc16(data, 8);  // 前8字节计算CRC

// 校验CRC16（数据包含CRC，以小端存储）
bool valid = tools::check_crc16(data, 10);  // 校验10字节（8字节数据+2字节CRC）
```

**在项目中的使用**:
- ✅ **通信模块**: 串口和CAN通信的数据校验
- 📍 **典型位置**:
  - `io/gimbal/gimbal.cpp` - 云台串口通信CRC校验
  - `io/dm_imu/dm_imu.cpp` - IMU串口通信CRC校验

**为什么需要它**:
- 检测数据传输错误
- 确保通信数据的可靠性
- 避免错误数据导致系统故障

---

### 13. RANSAC正弦拟合 (ransac_sine_fitter)

**文件**: `ransac_sine_fitter.hpp`, `ransac_sine_fitter.cpp`

**功能**: 使用RANSAC算法拟合正弦函数，用于BUFF目标预测。

**特点**:
- RANSAC算法抗噪声能力强
- 拟合正弦函数：`y = A * sin(ω*t + φ) + C`
- 自动检测内点（inliers）

**使用示例**:
```cpp
#include "tools/ransac_sine_fitter.hpp"

// 创建拟合器
tools::RansacSineFitter fitter(1000, 0.1, 0.1, 10.0);
// 参数：最大迭代次数、阈值、最小角频率、最大角频率

// 添加数据点
fitter.add_data(t1, v1);
fitter.add_data(t2, v2);
// ...

// 执行拟合
fitter.fit();

// 获取结果
auto result = fitter.best_result_;
double A = result.A;        // 振幅
double omega = result.omega; // 角频率
double phi = result.phi;    // 相位
double C = result.C;        // 偏移
int inliers = result.inliers; // 内点数量
```

**在项目中的使用**:
- ✅ **BUFF模块**: 预测旋转目标的位置
- 📍 **典型位置**:
  - `tasks/auto_buff/buff_predict.hpp` - BUFF目标预测

**为什么需要它**:
- BUFF目标做圆周运动，轨迹是正弦函数
- RANSAC算法可以处理噪声和异常值
- 拟合后可以预测目标未来位置

---

### 14. 线程池 (thread_pool)

**文件**: `thread_pool.hpp`

**功能**: 提供线程池和有序队列，用于多线程YOLO检测。

**特点**:
- 线程池管理多个工作线程
- 有序队列保证结果按帧顺序输出
- 支持批量创建YOLO实例

**使用示例**:
```cpp
#include "tools/thread_pool.hpp"

// 创建多个YOLO实例
auto yolos = tools::create_yolo11s(config_path, 4, false);  // 4个YOLO实例

// 使用线程池进行检测
// （具体使用方式见 tasks/auto_aim/multithread/）
```

**在项目中的使用**:
- ✅ **多线程检测**: 提高检测速度
- 📍 **典型位置**:
  - `tasks/auto_aim/multithread/mt_detector.cpp` - 多线程检测器

**为什么需要它**:
- YOLO检测是计算密集型任务
- 多线程可以充分利用多核CPU
- 有序队列确保结果顺序正确

---

## 使用建议

### 1. 新手入门路径

1. **先学基础工具**:
   - `logger` - 学会记录日志，方便调试
   - `exiter` - 理解程序退出机制
   - `yaml` - 学会读取配置

2. **再学数学工具**:
   - `math_tools` - 理解坐标转换、角度处理
   - `trajectory` - 理解弹道计算

3. **最后学高级工具**:
   - `extended_kalman_filter` - 理解状态估计
   - `thread_safe_queue` - 理解多线程编程

### 2. 常见使用模式

#### 模式1: 主程序模板
```cpp
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

int main() {
    tools::Exiter exiter;
    auto yaml = tools::load("configs/config.yaml");
    
    while (!exiter.exit()) {
        // 主循环
        tools::logger()->info("Processing...");
    }
}
```

#### 模式2: 多线程数据传递
```cpp
#include "tools/thread_safe_queue.hpp"

tools::ThreadSafeQueue<Data> queue(1000);

// 生产者线程
void producer() {
    Data data;
    queue.push(data);
}

// 消费者线程
void consumer() {
    Data data;
    queue.pop(data);
}
```

#### 模式3: 坐标转换
```cpp
#include "tools/math_tools.hpp"

// 四元数转欧拉角
Eigen::Quaterniond q;
Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0);

// 坐标转换
Eigen::Vector3d xyz(1, 2, 3);
Eigen::Vector3d ypd = tools::xyz2ypd(xyz);
```

---

## 常见问题

### Q1: 为什么需要线程安全队列？

**A**: 在多线程环境下，多个线程同时访问同一个数据结构会导致数据竞争。`ThreadSafeQueue` 使用互斥锁保护，确保同一时间只有一个线程可以访问队列，避免了数据竞争问题。

### Q2: logger 和 printf 有什么区别？

**A**: 
- `logger` 支持多级别日志（debug/info/warn/error）
- `logger` 可以同时输出到控制台和文件
- `logger` 线程安全，可以在多线程环境下使用
- `logger` 自动添加时间戳和日志级别

### Q3: 什么时候使用 Trajectory？

**A**: 当你需要计算弹丸的飞行轨迹时使用。例如：
- 计算瞄准角度（考虑重力下坠）
- 计算飞行时间（用于预测）
- 判断目标是否在射程内

### Q4: EKF 和普通卡尔曼滤波有什么区别？

**A**: 
- 普通KF只能处理线性系统
- EKF可以处理非线性系统（通过线性化）
- 我们的目标跟踪是非线性的（涉及角度、距离等），所以使用EKF

### Q5: 如何选择合适的工具？

**A**: 
- **需要记录日志** → 使用 `logger`
- **需要多线程通信** → 使用 `thread_safe_queue`
- **需要坐标转换** → 使用 `math_tools`
- **需要读取配置** → 使用 `yaml`
- **需要弹道计算** → 使用 `trajectory`
- **需要状态估计** → 使用 `extended_kalman_filter`

---

## 总结

`tools/` 目录是项目的基础设施，提供了：

1. **基础功能**: 日志、配置、退出控制
2. **并发支持**: 线程安全队列、线程池
3. **数学计算**: 坐标转换、弹道计算、滤波
4. **调试工具**: 记录、可视化、图像绘制

这些工具被项目的各个模块广泛使用，是理解整个系统的基础。建议新手先从基础工具开始学习，逐步深入理解高级工具的使用。

---

**最后更新**: 2025年  
**维护者**: 项目团队  
**反馈**: 如有问题或建议，请提交 Issue 或联系项目负责人

