# Src 目录详细文档

## 📋 目录

1. [目录概述](#目录概述)
2. [程序分类](#程序分类)
3. [核心程序详解](#核心程序详解)
4. [程序架构与数据流](#程序架构与数据流)
5. [在项目中的作用](#在项目中的作用)
6. [使用指南](#使用指南)

---

## 目录概述

`src` 目录包含机器人视觉系统的所有可执行程序入口点。这些程序实现了不同的功能模式（自瞄、打符、哨兵等），并提供了多种架构实现（单线程、多线程、MPC控制等）和调试工具。

### 文件列表

| 文件名 | 类型 | 功能 | 特点 |
|--------|------|------|------|
| `standard.cpp` | 标准程序 | 基础自瞄 | YOLO检测，单线程 |
| `mt_standard.cpp` | 标准程序 | 多线程自瞄 | 多线程检测，支持打符 |
| `uav.cpp` | 标准程序 | 无人机自瞄 | 支持自瞄和打符切换 |
| `sentry.cpp` | 哨兵程序 | 哨兵自瞄 | 全向感知，ROS2通信 |
| `sentry_bp.cpp` | 哨兵程序 | 哨兵备份 | 简化版哨兵 |
| `sentry_debug.cpp` | 调试程序 | 哨兵调试 | 带可视化调试 |
| `sentry_multithread.cpp` | 哨兵程序 | 多线程哨兵 | 全向感知+多线程 |
| `standard_mpc.cpp` | MPC程序 | MPC控制 | 模型预测控制 |
| `auto_aim_debug_mpc.cpp` | 调试程序 | MPC调试 | MPC可视化调试 |
| `auto_buff_debug.cpp` | 调试程序 | 打符调试 | 能量机关调试 |
| `uav_debug.cpp` | 调试程序 | 无人机调试 | 无人机可视化调试 |
| `mt_auto_aim_debug.cpp` | 调试程序 | 多线程调试 | 多线程可视化调试 |
| `hero.cpp` | 占位文件 | - | 空文件 |

---

## 程序分类

### 1. 标准程序（Standard）

**特点**: 基础自瞄功能，适用于标准步兵、英雄等兵种

#### `standard.cpp` - 标准自瞄程序

**功能**: 实现基础的自瞄功能

**实现思路**:
1. 初始化所有模块（相机、C板、检测器、跟踪器、瞄准器等）
2. 主循环：
   - 读取图像和IMU数据
   - 更新坐标系（通过IMU四元数）
   - 检测装甲板（YOLO）
   - 跟踪目标
   - 计算瞄准角度
   - 发送控制命令

**关键代码流程**:
```cpp
while (!exiter.exit()) {
    camera.read(img, t);                    // 读取图像
    q = cboard.imu_at(t - 1ms);             // 获取IMU数据
    solver.set_R_gimbal2world(q);           // 更新坐标系
    
    auto armors = detector.detect(img);     // 检测装甲板
    auto targets = tracker.track(armors, t); // 跟踪目标
    auto command = aimer.aim(targets, t, cboard.bullet_speed); // 瞄准
    
    cboard.send(command);                   // 发送命令
}
```

**在项目中的作用**: 作为最基础的自瞄程序，是其他程序的参考实现。

---

#### `mt_standard.cpp` - 多线程标准程序

**功能**: 多线程版本的标准自瞄，提高性能

**实现思路**:
1. **检测线程**: 独立线程进行图像采集和检测
   - 使用 `MultiThreadDetector` 进行异步检测
   - 通过队列传递检测结果

2. **主线程**: 处理跟踪、瞄准和打符
   - 从检测队列获取结果
   - 执行跟踪和瞄准
   - 支持打符模式切换

**关键架构**:
```cpp
// 检测线程
auto detect_thread = std::thread([&]() {
    while (!exiter.exit()) {
        if (mode == io::Mode::auto_aim) {
            camera.read(img, t);
            detector.push(img, t);  // 异步检测
        }
    }
});

// 主线程
while (!exiter.exit()) {
    auto [img, armors, t] = detector.debug_pop();  // 获取检测结果
    // 处理跟踪和瞄准
}
```

**优势**:
- 检测和跟踪并行，提高帧率
- 检测线程可以持续工作，不阻塞主线程

**在项目中的作用**: 提供高性能的多线程实现，适用于对实时性要求高的场景。

---

#### `uav.cpp` - 无人机程序

**功能**: 支持自瞄和打符两种模式切换

**实现思路**:
1. 根据 `cboard.mode` 切换模式
2. **自瞄模式** (`auto_aim` 或 `outpost`):
   - 使用传统CV检测器（`Detector`）
   - 执行标准自瞄流程

3. **打符模式** (`small_buff` 或 `big_buff`):
   - 使用能量机关检测器
   - 执行打符流程

**关键代码**:
```cpp
if (mode == io::Mode::auto_aim || mode == io::Mode::outpost) {
    // 自瞄逻辑
    auto armors = detector.detect(img);
    auto targets = tracker.track(armors, t);
    auto command = aimer.aim(targets, t, cboard.bullet_speed);
    cboard.send(command);
}
else if (mode == io::Mode::small_buff || mode == io::Mode::big_buff) {
    // 打符逻辑
    auto power_runes = buff_detector.detect(img);
    buff_solver.solve(power_runes);
    // ...
}
```

**在项目中的作用**: 提供多模式支持，适用于需要切换不同任务的场景。

---

### 2. 哨兵程序（Sentry）

**特点**: 专门为哨兵机器人设计，支持全向感知和ROS2通信

#### `sentry.cpp` - 哨兵主程序

**功能**: 完整的哨兵自瞄系统，支持全向感知

**实现思路**:
1. **初始化多相机**:
   - 主相机（`camera`）
   - 后置相机（`back_camera`）
   - USB相机（`usbcam1`, `usbcam2`）

2. **全向感知集成**:
   - 使用 `omniperception::Decider` 进行决策
   - 获取敌方状态（无敌状态）
   - 过滤装甲板
   - 设置优先级

3. **目标切换逻辑**:
   - 如果跟踪器状态为 `lost`，使用全向感知决策
   - 否则使用标准瞄准

4. **ROS2通信**:
   - 发布目标信息
   - 订阅敌方状态

**关键代码流程**:
```cpp
// 自瞄核心逻辑
solver.set_R_gimbal2world(q);
auto armors = yolo.detect(img);

// 全向感知处理
decider.get_invincible_armor(ros2.subscribe_enemy_status());
decider.armor_filter(armors);
decider.set_priority(armors);

auto targets = tracker.track(armors, timestamp);

// 决策
if (tracker.state() == "lost")
    command = decider.decide(yolo, gimbal_pos, usbcam1, usbcam2, back_camera);
else
    command = aimer.aim(targets, timestamp, cboard.bullet_speed, cboard.shoot_mode);

// ROS2通信
ros2.publish(decider.get_target_info(armors, targets));
```

**在项目中的作用**: 作为哨兵机器人的核心程序，实现全向感知和智能决策。

---

#### `sentry_bp.cpp` - 哨兵备份程序

**功能**: 简化版哨兵程序，作为备份

**实现思路**:
- 与 `sentry.cpp` 类似，但使用后置相机而非USB相机
- 简化了全向感知逻辑

**在项目中的作用**: 提供备用方案，当主程序出现问题时使用。

---

#### `sentry_debug.cpp` - 哨兵调试程序

**功能**: 带可视化调试功能的哨兵程序

**实现思路**:
- 在 `sentry.cpp` 基础上添加调试功能
- 可视化显示：
  - 装甲板重投影（绿色）
  - 瞄准点重投影（红色/蓝色）
  - EKF状态数据
  - 卡方检验数据（NIS、NEES等）

**调试数据输出**:
```cpp
// EKF状态
data["x"], data["vx"], data["y"], data["vy"], data["z"], data["vz"]
data["a"], data["w"], data["r"], data["l"], data["h"]

// 卡方检验
data["nis"], data["nees"], data["nis_fail"], data["nees_fail"]
data["residual_yaw"], data["residual_pitch"], etc.
```

**在项目中的作用**: 用于调试和优化哨兵系统，分析跟踪和瞄准性能。

---

#### `sentry_multithread.cpp` - 多线程哨兵程序

**功能**: 结合多线程和全向感知的哨兵程序

**实现思路**:
1. 使用 `omniperception::Perceptron` 进行多相机感知
2. 获取检测队列 `detection_queue`
3. 跟踪器支持全向感知目标切换：
   ```cpp
   auto [switch_target, targets] = tracker.track(detection_queue, armors, timestamp);
   ```

4. **状态处理**:
   - `switching`: 处理目标切换命令
   - `lost`: 使用全向感知决策
   - `tracking`: 标准瞄准

**在项目中的作用**: 提供最高性能的哨兵实现，结合多线程和全向感知。

---

### 3. MPC控制程序（Model Predictive Control）

**特点**: 使用模型预测控制进行云台控制，提供更平滑的控制

#### 什么是MPC？

**MPC（Model Predictive Control，模型预测控制）** 是一种先进的控制算法，其核心思想是：

1. **预测未来**: 根据当前状态和系统模型，预测未来一段时间内的系统行为
2. **优化控制**: 在预测的时间窗口内，优化控制序列，使系统尽可能接近期望轨迹
3. **滚动优化**: 每个控制周期都重新计算最优控制序列，但只执行第一步
4. **反馈校正**: 根据实际测量值更新状态，在下个周期重新优化

**MPC vs 传统控制**:

| 特性 | 传统控制（PID/位置控制） | MPC控制 |
|------|------------------------|---------|
| 控制方式 | 仅位置控制 | 位置 + 速度 + 加速度 |
| 预测能力 | 无 | 预测未来轨迹 |
| 约束处理 | 困难 | 可处理约束（加速度限制等） |
| 平滑性 | 中等 | 高（考虑未来状态） |
| 计算复杂度 | 低 | 高（需要求解优化问题） |

#### 在本项目中的MPC实现

**核心组件**:
- **Planner**: MPC规划器，负责生成控制轨迹
- **TinyMPC**: 轻量级MPC求解器（求解耗时 < 1ms）
- **双轴独立控制**: Yaw轴和Pitch轴分别使用独立的MPC控制器

**工作流程**:

1. **生成参考轨迹**:
   ```cpp
   // 预测目标未来1秒的轨迹（HORIZON = 100，DT = 0.01s）
   Trajectory traj = get_trajectory(target, yaw0, bullet_speed);
   // traj包含: [yaw, yaw_vel, pitch, pitch_vel] × 100个时间步
   ```

2. **MPC优化求解**:
   ```cpp
   // Yaw轴MPC求解
   yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);  // 参考轨迹
   tiny_solve(yaw_solver_);  // 求解最优控制序列
   
   // Pitch轴MPC求解
   pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
   tiny_solve(pitch_solver_);
   ```

3. **提取控制命令**:
   ```cpp
   // 取中间时刻（HALF_HORIZON = 50）的控制量
   plan.yaw = yaw_solver_->work->x(0, HALF_HORIZON);      // 位置
   plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON); // 速度
   plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);  // 加速度
   ```

**系统模型**:
```
状态: [位置, 速度]
控制: [加速度]

状态转移:
  x[k+1] = A * x[k] + B * u[k]
  
其中:
  A = [[1, DT],    (位置 = 位置 + 速度*DT)
       [0, 1]]     (速度 = 速度)
       
  B = [[0],        (位置不受加速度直接影响)
       [DT]]       (速度 = 速度 + 加速度*DT)
```

**约束条件**:
- 加速度限制: `-max_acc <= u <= max_acc`
- 状态无限制（实际由物理限制保证）

**代价函数**:
```
J = Σ (x[k] - x_ref[k])^T * Q * (x[k] - x_ref[k]) 
  + Σ u[k]^T * R * u[k]
```
- 第一项: 跟踪误差（使实际轨迹接近参考轨迹）
- 第二项: 控制代价（使控制量尽可能小，更平滑）

#### `standard_mpc.cpp` - 标准MPC控制程序

**功能**: 使用MPC控制器进行云台控制

**实现思路**:
1. **初始化MPC规划器**:
   - 使用 `auto_aim::Planner` 进行MPC规划
   - 使用 `io::Gimbal` 进行云台控制（而非C板）
   - 分别设置Yaw和Pitch的MPC求解器

2. **规划线程**:
   - 独立线程执行MPC规划（10ms周期）
   - 从目标队列获取目标
   - 计算控制命令（位置、速度、加速度）
   - 发送到云台

3. **主线程**:
   - 检测和跟踪目标
   - 将目标推送到队列

**关键架构**:
```cpp
// 规划线程（10ms周期）
auto plan_thread = std::thread([&]() {
    while (!quit) {
        if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
            auto target = target_queue.front();
            auto gs = gimbal.state();
            auto plan = planner.plan(target, gs.bullet_speed);
            
            // 发送位置、速度、加速度
            gimbal.send(plan.control, plan.fire, 
                       plan.yaw, plan.yaw_vel, plan.yaw_acc,
                       plan.pitch, plan.pitch_vel, plan.pitch_acc);
        }
        std::this_thread::sleep_for(10ms);
    }
});

// 主线程（检测和跟踪）
auto armors = yolo.detect(img);
auto targets = tracker.track(armors, t);
if (!targets.empty())
    target_queue.push(targets.front());
```

**MPC优势**:
- **平滑控制**: 考虑未来状态，控制更平滑，减少抖动
- **延迟补偿**: 可以预测和补偿系统延迟
- **多级控制**: 提供位置、速度、加速度三级控制，响应更快
- **约束处理**: 自动满足加速度限制等约束
- **提前减速**: 在接近目标时提前减速，提高精度

**射击决策**:
```cpp
// 检查未来时刻（HALF_HORIZON + shoot_offset_）的跟踪误差
plan.fire = std::hypot(
    traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
    traj(2, HALF_HORIZON + shoot_offset_) - pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)
) < fire_thresh_;
```
通过预测未来时刻的跟踪误差来判断是否可以射击，比仅看当前位置更准确。

**在项目中的作用**: 提供高级控制方案，适用于对控制精度要求高的场景，特别是在跟踪高速运动目标时。

---

#### `auto_aim_debug_mpc.cpp` - MPC调试程序

**功能**: MPC控制的可视化调试

**实现思路**:
- 在 `standard_mpc.cpp` 基础上添加调试功能
- 可视化显示：
  - 所有装甲板重投影（绿色）
  - 瞄准点重投影（红色）
- 数据输出：
  - 云台状态（yaw, pitch, 速度）
  - 目标状态（yaw, pitch, 速度）
  - 规划结果（位置、速度、加速度）
  - 射击状态

**在项目中的作用**: 用于调试和优化MPC控制器参数。

---

### 4. 调试程序（Debug）

**特点**: 提供详细的可视化和数据输出，用于调试和优化

#### `auto_buff_debug.cpp` - 打符调试程序

**功能**: 能量机关（打符）任务的调试程序

**实现思路**:
1. 检测能量机关（`Buff_Detector`）
2. 解算位置（`Buff_Solver`）
3. 跟踪目标（`SmallTarget` 或 `BigTarget`）
4. 计算瞄准角度（`Buff_Aimer`）

**调试输出**:
- 能量机关原始观测数据（yaw, pitch, distance）
- EKF状态（R_yaw, R_pitch, R_dis, yaw, angle, speed等）
- 重投影可视化（当前帧绿色，预测红色）
- 云台响应数据

**在项目中的作用**: 用于调试和优化打符算法。

---

#### `uav_debug.cpp` - 无人机调试程序

**功能**: 无人机自瞄的调试程序

**实现思路**:
- 在 `uav.cpp` 基础上添加调试功能
- 可视化显示：
  - 装甲板重投影
  - 瞄准点重投影
- 数据输出：
  - 装甲板原始观测
  - EKF状态
  - 卡方检验数据

**在项目中的作用**: 用于调试无人机自瞄系统。

---

#### `mt_auto_aim_debug.cpp` - 多线程调试程序

**功能**: 多线程自瞄的调试程序

**实现思路**:
- 在 `mt_standard.cpp` 基础上添加调试功能
- 使用多线程检测器
- 输出详细的调试数据

**在项目中的作用**: 用于调试多线程自瞄系统。

---

## 程序架构与数据流

### 标准程序架构

```
┌─────────────┐
│   Camera    │──→ 图像
└─────────────┘
      │
      ↓
┌─────────────┐
│  Detector   │──→ Armors
└─────────────┘
      │
      ↓
┌─────────────┐
│  Tracker    │──→ Targets
└─────────────┘
      │
      ↓
┌─────────────┐
│   Aimer     │──→ Command
└─────────────┘
      │
      ↓
┌─────────────┐
│   Shooter   │──→ Command (with shoot)
└─────────────┘
      │
      ↓
┌─────────────┐
│   CBoard    │──→ 发送到硬件
└─────────────┘
```

### 多线程程序架构

```
┌─────────────────┐
│  Detect Thread  │
│  - 图像采集      │
│  - 检测         │
└─────────────────┘
         │
         ↓ Queue
┌─────────────────┐
│  Main Thread    │
│  - 跟踪         │
│  - 瞄准         │
│  - 发送命令      │
└─────────────────┘
```

### MPC程序架构

```
┌─────────────────┐
│  Main Thread    │
│  - 检测         │
│  - 跟踪         │
│  - 推送目标      │
└─────────────────┘
         │
         ↓ Queue
┌─────────────────┐
│  Plan Thread    │
│  - MPC规划      │
│  - 发送命令      │
└─────────────────┘
```

### 哨兵程序架构

```
┌─────────────┐
│   Camera    │
│  (Multiple) │
└─────────────┘
      │
      ↓
┌─────────────┐     ┌─────────────┐
│    YOLO     │────→│   Decider   │
└─────────────┘     └─────────────┘
      │                   │
      ↓                   ↓
┌─────────────┐     ┌─────────────┐
│  Tracker    │     │  Perceptron │
└─────────────┘     └─────────────┘
      │                   │
      └─────────┬─────────┘
                ↓
         ┌─────────────┐
         │   Aimer     │
         └─────────────┘
                │
                ↓
         ┌─────────────┐
         │   ROS2     │
         └─────────────┘
```

---

## 在项目中的作用

### 1. 程序入口点

`src` 目录中的所有程序都是可执行程序的入口点（`main`函数），它们：

- **集成所有模块**: 将IO模块、任务模块、工具模块组合在一起
- **实现业务逻辑**: 根据不同的应用场景实现不同的控制流程
- **提供可执行文件**: 编译后生成可执行文件，直接运行

### 2. 不同场景的适配

| 程序 | 适用场景 | 特点 |
|------|---------|------|
| `standard.cpp` | 标准步兵/英雄 | 基础自瞄，简单可靠 |
| `mt_standard.cpp` | 高性能需求 | 多线程，高帧率 |
| `uav.cpp` | 无人机 | 多模式切换 |
| `sentry.cpp` | 哨兵 | 全向感知，ROS2 |
| `standard_mpc.cpp` | 高精度控制 | MPC控制 |

### 3. 调试和优化工具

调试程序提供：
- **可视化**: 重投影显示、状态可视化
- **数据输出**: JSON格式的详细数据，用于分析
- **性能监控**: 帧率、延迟等指标

### 4. 代码复用

所有程序都：
- 使用相同的底层模块（`auto_aim`, `auto_buff`等）
- 遵循相同的架构模式
- 可以轻松切换和组合功能

---

## 使用指南

### 编译

```bash
cd build
cmake ..
make standard        # 编译标准程序
make mt_standard     # 编译多线程程序
make sentry          # 编译哨兵程序
make standard_mpc    # 编译MPC程序
# ... 其他程序
```

### 运行

```bash
# 标准程序
./standard configs/standard3.yaml

# 多线程程序
./mt_standard --config-path=configs/standard3.yaml

# 哨兵程序
./sentry configs/sentry.yaml

# MPC程序
./standard_mpc configs/sentry.yaml
```

### 选择程序

**根据需求选择**:

1. **基础自瞄**: `standard.cpp`
2. **高性能**: `mt_standard.cpp`
3. **多模式**: `uav.cpp`
4. **哨兵**: `sentry.cpp` 或 `sentry_multithread.cpp`
5. **高精度控制**: `standard_mpc.cpp`
6. **调试**: 对应的 `*_debug.cpp` 程序

### 配置

所有程序都通过YAML配置文件进行配置：

```yaml
# configs/standard3.yaml
camera_name: "hikrobot_gige"
enemy_color: "red"
# ... 其他参数
```

### 调试

使用调试程序时：
1. 运行对应的 `*_debug.cpp` 程序
2. 查看可视化窗口（重投影显示）
3. 查看数据输出（JSON格式）
4. 按 'q' 退出

---

## 程序对比

### 单线程 vs 多线程

| 特性 | 单线程 | 多线程 |
|------|--------|--------|
| 实现复杂度 | 简单 | 复杂 |
| 性能 | 中等 | 高 |
| 延迟 | 较高 | 较低 |
| 适用场景 | 基础应用 | 高性能需求 |

### 传统控制 vs MPC控制

| 特性 | 传统控制 | MPC控制 |
|------|---------|---------|
| 控制方式 | 位置控制 | 位置+速度+加速度 |
| 平滑性 | 中等 | 高 |
| 响应速度 | 快 | 中等 |
| 实现复杂度 | 简单 | 复杂 |
| 适用场景 | 一般应用 | 高精度需求 |

### 标准程序 vs 哨兵程序

| 特性 | 标准程序 | 哨兵程序 |
|------|---------|---------|
| 相机数量 | 1个 | 多个 |
| 全向感知 | 否 | 是 |
| ROS2通信 | 否 | 是 |
| 决策逻辑 | 简单 | 复杂 |
| 适用场景 | 步兵/英雄 | 哨兵 |

---

## 开发建议

### 添加新程序

1. **复制现有程序**: 从最接近的程序开始
2. **修改功能**: 根据需要添加或删除功能
3. **更新CMakeLists.txt**: 添加新的可执行文件
4. **测试**: 使用调试程序进行测试

### 优化性能

1. **使用多线程**: 对于高性能需求，使用多线程版本
2. **减少延迟**: 优化检测和跟踪算法
3. **使用MPC**: 对于高精度需求，使用MPC控制

### 调试技巧

1. **使用调试程序**: 运行 `*_debug.cpp` 程序
2. **查看可视化**: 检查重投影是否正确
3. **分析数据**: 查看JSON输出，分析性能
4. **日志输出**: 使用 `tools::logger()` 输出关键信息

---

## 常见问题

### 1. 如何选择程序？

**答**: 根据应用场景选择：
- 标准步兵/英雄 → `standard.cpp`
- 需要高性能 → `mt_standard.cpp`
- 哨兵 → `sentry.cpp`
- 需要高精度控制 → `standard_mpc.cpp`

### 2. 如何添加新功能？

**答**: 
1. 在对应的任务模块中添加功能
2. 在程序中使用新功能
3. 测试和调试

### 3. 如何调试程序？

**答**: 
1. 使用对应的 `*_debug.cpp` 程序
2. 查看可视化窗口
3. 分析数据输出
4. 使用日志输出

### 4. 多线程程序的优势？

**答**: 
- 检测和跟踪并行，提高帧率
- 减少延迟
- 更好的实时性

### 5. MPC控制的优势？

**答**: 
- 更平滑的控制
- 可以预测未来状态
- 提供速度、加速度控制
- 更好的跟踪性能

---

## 总结

`src` 目录包含了机器人视觉系统的所有可执行程序，它们：

1. **实现不同场景**: 标准自瞄、哨兵、无人机等
2. **提供不同架构**: 单线程、多线程、MPC控制
3. **支持调试**: 提供详细的调试工具
4. **代码复用**: 共享底层模块，易于维护

选择合适的程序对于系统的性能和可靠性至关重要。根据实际需求选择最合适的程序，并在需要时使用调试程序进行优化。

---

**最后更新**: 2025年

