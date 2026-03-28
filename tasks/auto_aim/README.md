# Auto Aim 模块详细文档

## 📋 目录

1. [模块概述](#模块概述)
2. [系统架构](#系统架构)
3. [核心组件详解](#核心组件详解)
4. [数据流与处理流程](#数据流与处理流程)
5. [关键算法与实现思路](#关键算法与实现思路)
6. [在项目中的作用](#在项目中的作用)
7. [使用示例](#使用示例)

---

## 模块概述

`auto_aim` 模块是机器人视觉系统中的核心自动瞄准模块，负责从相机图像中检测、跟踪敌方装甲板，并计算瞄准角度以实现自动射击。该模块采用传统计算机视觉与深度学习相结合的方法，实现了高精度的目标检测、跟踪和预测。

### 主要功能

- **装甲板检测**：从图像中检测敌方装甲板（支持传统CV方法和YOLO深度学习方法）
- **目标跟踪**：使用扩展卡尔曼滤波（EKF）跟踪目标运动状态
- **姿态解算**：通过PnP算法计算装甲板在3D空间中的位置和姿态
- **弹道预测**：考虑子弹飞行时间和目标运动，预测瞄准点
- **射击决策**：根据瞄准精度和稳定性决定是否射击

---

## 系统架构

### 模块依赖关系

```
相机图像 (cv::Mat)
    ↓
[Detector/YOLO] → 检测装甲板 → std::list<Armor>
    ↓
[Tracker] → 跟踪目标 → std::list<Target>
    ↓
[Aimer] → 计算瞄准角度 → io::Command
    ↓
[Shooter] → 射击决策 → bool
    ↓
发送控制命令到云台
```

### 核心类关系

```
Solver (坐标变换、PnP求解)
    ↑
    ├── Detector (使用Solver进行坐标变换)
    ├── Tracker (使用Solver初始化Target)
    └── Aimer (使用Solver进行弹道计算)

Target (EKF状态估计)
    ↑
    └── Tracker (管理Target生命周期)

Armor (数据结构)
    ↑
    ├── Detector (生成Armor)
    ├── Tracker (匹配Armor到Target)
    └── Aimer (使用Armor计算瞄准点)
```

---

## 核心组件详解

### 1. Detector (检测器)

**文件**: `detector.hpp`, `detector.cpp`

**功能**: 从图像中检测装甲板

**实现思路**:

1. **图像预处理**
   - 彩色图转灰度图
   - 二值化处理（阈值分割）
   - 轮廓检测

2. **灯条检测**
   - 从轮廓中提取旋转矩形
   - 几何约束过滤（角度、长宽比、长度）
   - 颜色识别（红/蓝）

3. **装甲板匹配**
   - 配对同色灯条
   - 几何约束验证（宽高比、矩形度误差）
   - 提取装甲板图案（ROI）

4. **分类识别**
   - 使用分类器识别装甲板编号（1-5、哨兵、前哨站、基地）
   - 判断装甲板类型（大/小）

5. **去重处理**
   - 检测共用灯条的情况
   - 保留置信度更高或ROI更小的装甲板

**关键参数**:
- `threshold_`: 二值化阈值
- `max_angle_error_`: 灯条角度误差阈值
- `min_lightbar_ratio_`, `max_lightbar_ratio_`: 灯条长宽比范围
- `min_armor_ratio_`, `max_armor_ratio_`: 装甲板宽高比范围
- `min_confidence_`: 分类器置信度阈值

**特殊功能**:
- `lightbar_points_corrector()`: 使用PCA回归优化灯条角点位置（参考FYT2024）
- `detect(Armor&, cv::Mat&)`: 在ROI内重新检测装甲板，用于精确定位

**在项目中的作用**: 作为视觉感知的第一环节，将图像转换为结构化的装甲板数据，为后续跟踪和瞄准提供输入。

---

### 2. Tracker (跟踪器)

**文件**: `tracker.hpp`, `tracker.cpp`

**功能**: 跟踪目标装甲板，管理目标状态

**实现思路**:

1. **状态机设计**
   - `lost`: 丢失目标
   - `detecting`: 检测中（需要连续检测N帧才进入tracking）
   - `tracking`: 跟踪中
   - `temp_lost`: 临时丢失（允许短暂丢失）
   - `switching`: 切换目标（全向感知场景）

2. **目标选择策略**
   - 按优先级排序（英雄>工程>步兵>哨兵>前哨站>基地）
   - 优先选择靠近图像中心的装甲板
   - 过滤非敌方颜色装甲板

3. **目标初始化**
   - 根据兵种类型设置不同的EKF初始化参数
   - 平衡步兵（3/4/5号大装甲）使用特殊参数
   - 前哨站和基地使用不同的半径和装甲数量

4. **目标更新**
   - 预测目标状态（EKF predict）
   - 匹配当前帧检测到的装甲板
   - 更新目标状态（EKF update）

5. **发散检测**
   - 检测EKF状态是否发散（半径、长度是否合理）
   - 检测NIS（归一化创新平方）失败率
   - 发散时重置为lost状态

**关键参数**:
- `min_detect_count_`: 进入tracking状态需要连续检测的帧数
- `max_temp_lost_count_`: 临时丢失的最大帧数
- `outpost_max_temp_lost_count_`: 前哨站临时丢失的最大帧数

**在项目中的作用**: 实现目标的稳定跟踪，即使在短暂遮挡或检测失败的情况下也能保持跟踪，为瞄准提供连续的目标状态。

---

### 3. Target (目标)

**文件**: `target.hpp`, `target.cpp`

**功能**: 使用扩展卡尔曼滤波（EKF）估计目标运动状态

**实现思路**:

1. **状态向量设计**
   ```
   x = [x, vx, y, vy, z, vz, a, w, r, l, h]
   ```
   - `x, y, z`: 旋转中心位置（世界坐标系）
   - `vx, vy, vz`: 线速度
   - `a`: 当前装甲板角度（yaw）
   - `w`: 角速度
   - `r`: 旋转半径（短轴）
   - `l`: 长短轴差（r2 - r1）
   - `h`: 高度差（z2 - z1）

2. **运动模型**
   - 使用匀速运动模型（CV模型）
   - 状态转移矩阵F考虑时间步长dt
   - 过程噪声Q使用分段白噪声模型（Piecewise White Noise）

3. **观测模型**
   - 观测量为 `[yaw, pitch, distance, angle]`（球坐标系）
   - 非线性观测函数h：从状态向量计算装甲板3D位置，再转换为球坐标
   - 雅可比矩阵H：观测函数对状态向量的偏导数

4. **装甲板匹配**
   - 根据角度误差匹配当前检测到的装甲板到EKF预测的装甲板
   - 选择距离最近且角度误差最小的装甲板
   - 检测装甲板跳变（jumped）

5. **收敛判断**
   - 非前哨站：更新3次以上且未发散
   - 前哨站：更新10次以上且未发散

**关键算法**:
- `h_armor_xyz()`: 根据状态向量和装甲板ID计算装甲板3D位置
- `h_jacobian()`: 计算观测雅可比矩阵
- `armor_xyza_list()`: 生成所有装甲板的3D位置和角度列表

**在项目中的作用**: 提供目标运动的平滑估计和预测，即使目标被短暂遮挡也能预测其位置，为瞄准提供准确的目标状态。

---

### 4. Solver (求解器)

**文件**: `solver.hpp`, `solver.cpp`

**功能**: 坐标变换、PnP求解、重投影

**实现思路**:

1. **坐标系转换链**
   ```
   相机坐标系 (camera)
       ↓ R_camera2gimbal, t_camera2gimbal
   云台坐标系 (gimbal)
       ↓ R_gimbal2world
   世界坐标系 (world)
   ```

2. **PnP求解**
   - 使用 `cv::solvePnP` 求解装甲板在相机坐标系中的位姿
   - 已知3D点（装甲板角点）和2D点（图像像素）
   - 使用IPPE算法（快速且稳定）

3. **坐标变换**
   - 相机坐标系 → 云台坐标系
   - 云台坐标系 → 世界坐标系（通过IMU四元数）
   - 计算欧拉角（yaw, pitch, roll）

4. **Yaw优化**
   - 对于非平衡步兵，进行yaw角度优化
   - 在±70度范围内搜索最优yaw
   - 使用重投影误差作为代价函数
   - 平衡步兵不做优化（pitch假设不成立）

5. **重投影**
   - `reproject_armor()`: 根据3D位置和yaw角度重投影到图像
   - `world2pixel()`: 世界坐标点投影到像素坐标

**关键参数**:
- `camera_matrix_`: 相机内参矩阵
- `distort_coeffs_`: 畸变系数
- `R_camera2gimbal_`, `t_camera2gimbal_`: 相机到云台的变换
- `R_gimbal2imubody_`: 云台到IMU本体的旋转

**在项目中的作用**: 将2D图像坐标转换为3D世界坐标，建立图像与物理世界的对应关系，是所有后续计算的基础。

---

### 5. Aimer (瞄准器)

**文件**: `aimer.hpp`, `aimer.cpp`

**功能**: 计算瞄准角度，考虑弹道和飞行时间

**实现思路**:

1. **弹道预测迭代**
   - 初始预测：预测目标在 `timestamp + delay_time` 的位置
   - 迭代求解：考虑子弹飞行时间，迭代预测目标位置
   - 收敛条件：相邻两次飞行时间差 < 0.001s
   - 最多迭代10次

2. **瞄准点选择**
   - 如果目标未跳变：选择当前装甲板
   - 非小陀螺模式：选择在可射击范围内的装甲板（±60度）
   - 小陀螺模式：选择"来"装甲板（coming angle内）且"去"装甲板（leaving angle外）
   - 锁定模式：防止在两个45度装甲板之间来回切换

3. **弹道计算**
   - 使用 `tools::Trajectory` 计算弹道
   - 考虑重力、空气阻力
   - 计算pitch角度

4. **角度补偿**
   - `yaw_offset_`: yaw角度补偿
   - `pitch_offset_`: pitch角度补偿
   - 支持左右发射模式的不同补偿

**关键参数**:
- `yaw_offset_`, `pitch_offset_`: 角度补偿
- `comming_angle_`, `leaving_angle_`: 小陀螺模式的角度阈值
- `high_speed_delay_time_`, `low_speed_delay_time_`: 高速/低速延迟时间
- `decision_speed_`: 判断高速/低速的阈值

**在项目中的作用**: 将目标3D位置转换为云台控制角度，考虑弹道和飞行时间，实现精确瞄准。

---

### 6. Shooter (射击器)

**文件**: `shooter.hpp`, `shooter.cpp`

**功能**: 决定是否射击

**实现思路**:

1. **射击条件**
   - 控制命令有效
   - 目标列表非空
   - 自动射击模式开启

2. **稳定性判断**
   - 根据距离选择容差（远距离用second_tolerance，近距离用first_tolerance）
   - 检查yaw角度变化是否小于容差
   - 检查云台位置与命令角度是否匹配
   - 检查瞄准点是否有效

3. **防突变**
   - 如果command.yaw突变（> 2倍容差），不射击
   - 防止因检测错误导致的误射

**关键参数**:
- `first_tolerance_`, `second_tolerance_`: 角度容差
- `judge_distance_`: 判断使用哪个容差的距离阈值
- `auto_fire_`: 是否自动射击

**在项目中的作用**: 确保只在瞄准稳定时射击，提高命中率，避免误射。

---

### 7. Classifier (分类器)

**文件**: `classifier.hpp`, `classifier.cpp`

**功能**: 识别装甲板编号（1-5、哨兵、前哨站、基地）

**实现思路**:

1. **图像预处理**
   - 提取装甲板ROI（32x32）
   - 转为灰度图
   - 归一化到[0, 1]

2. **模型推理**
   - 支持OpenCV DNN和OpenVINO两种后端
   - OpenVINO优化为LATENCY模式（低延迟）

3. **后处理**
   - Softmax归一化
   - 选择置信度最高的类别

**在项目中的作用**: 识别装甲板编号，用于目标优先级排序和跟踪匹配。

---

### 8. Armor (装甲板数据结构)

**文件**: `armor.hpp`, `armor.cpp`

**功能**: 存储装甲板的所有信息

**数据结构**:

```cpp
struct Armor {
    // 图像信息
    Color color;                    // 颜色（红/蓝）
    Lightbar left, right;           // 左右灯条
    cv::Point2f center;             // 中心点
    std::vector<cv::Point2f> points;// 四个角点
    cv::Mat pattern;                // 装甲板图案（ROI）
    
    // 几何特征
    double ratio;                    // 宽高比
    double side_ratio;               // 灯条长度比
    double rectangular_error;        // 矩形度误差
    
    // 识别结果
    ArmorType type;                  // 大/小装甲
    ArmorName name;                  // 编号
    double confidence;                // 置信度
    ArmorPriority priority;           // 优先级
    
    // 3D信息
    Eigen::Vector3d xyz_in_gimbal;   // 云台坐标系位置
    Eigen::Vector3d xyz_in_world;    // 世界坐标系位置
    Eigen::Vector3d ypr_in_world;    // 世界坐标系欧拉角
    Eigen::Vector3d ypd_in_world;    // 世界坐标系球坐标
};
```

**构造函数**:
- 传统CV方法：从两个灯条构造
- YOLO方法：从检测框和关键点构造
- 支持ROI偏移（用于多尺度检测）

**在项目中的作用**: 作为整个系统的数据载体，连接检测、跟踪、瞄准各个环节。

---

### 9. YOLO (深度学习检测器)

**文件**: `yolo.hpp`, `yolo.cpp`, `yolos/`

**功能**: 使用YOLO模型进行装甲板检测

**实现思路**:

1. **多版本支持**
   - YOLOv5: 传统YOLO架构
   - YOLOv8: Ultralytics版本
   - YOLO11: 最新版本

2. **接口设计**
   - 使用策略模式，通过配置文件选择模型
   - 统一的接口：`detect()`, `postprocess()`

3. **后处理**
   - NMS（非极大值抑制）
   - 关键点解码
   - 坐标转换

**在项目中的作用**: 提供基于深度学习的检测方法，相比传统CV方法更鲁棒，特别是在复杂环境下。

---

### 10. Voter (投票器)

**文件**: `voter.hpp`, `voter.cpp`

**功能**: 统计装甲板类型投票

**实现思路**:
- 维护一个计数数组
- 根据颜色、名称、类型索引
- 用于多帧投票决策

**在项目中的作用**: 在多帧检测中统计装甲板类型，提高识别稳定性。

---

## 数据流与处理流程

### 完整处理流程

```
1. 图像采集
   camera.read(img, timestamp)
   
2. IMU数据获取
   q = cboard.imu_at(timestamp)
   solver.set_R_gimbal2world(q)
   
3. 装甲板检测
   armors = detector.detect(img)
   // 或
   armors = yolo.detect(img)
   
4. 目标跟踪
   targets = tracker.track(armors, timestamp)
   // Tracker内部：
   //   - 过滤非敌方装甲板
   //   - 按优先级排序
   //   - 状态机管理
   //   - EKF预测和更新
   
5. 瞄准计算
   command = aimer.aim(targets, timestamp, bullet_speed)
   // Aimer内部：
   //   - 预测目标未来位置
   //   - 迭代计算弹道
   //   - 选择瞄准点
   //   - 计算yaw和pitch角度
   
6. 射击决策
   command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos)
   
7. 发送命令
   cboard.send(command)
```

### 坐标系转换流程

```
图像像素坐标 (u, v)
    ↓ solvePnP
相机坐标系 (x_c, y_c, z_c)
    ↓ R_camera2gimbal, t_camera2gimbal
云台坐标系 (x_g, y_g, z_g)
    ↓ R_gimbal2world (从IMU四元数计算)
世界坐标系 (x_w, y_w, z_w)
    ↓ EKF预测
未来时刻世界坐标
    ↓ 弹道计算
瞄准角度 (yaw, pitch)
```

---

## 关键算法与实现思路

### 1. 扩展卡尔曼滤波（EKF）

**应用场景**: 目标运动状态估计

**状态向量**: 11维（位置、速度、角度、角速度、几何参数）

**预测步骤**:
```cpp
x_prior = F * x
P_prior = F * P * F^T + Q
```

**更新步骤**:
```cpp
K = P_prior * H^T * (H * P_prior * H^T + R)^(-1)
x = x_prior + K * (z - h(x_prior))
P = (I - K * H) * P_prior
```

**关键点**:
- 非线性观测函数h需要雅可比矩阵H
- 角度需要特殊处理（limit_rad）
- 过程噪声Q根据目标类型调整（前哨站vs普通目标）

### 2. PnP求解

**应用场景**: 从2D图像点计算3D位姿

**算法**: IPPE (Infinitesimal Plane-based Pose Estimation)

**输入**:
- 3D点（装甲板角点，已知尺寸）
- 2D点（图像像素坐标）

**输出**:
- 旋转向量rvec
- 平移向量tvec

**优化**: 对非平衡步兵进行yaw角度优化，搜索最优角度使重投影误差最小

### 3. 弹道预测迭代

**问题**: 子弹飞行时间未知，需要迭代求解

**算法**:
```
1. 预测目标在 t + delay_time 的位置
2. 计算弹道和飞行时间 fly_time
3. 预测目标在 t + delay_time + fly_time 的位置
4. 重新计算弹道和飞行时间
5. 重复直到收敛（|fly_time_new - fly_time_old| < 0.001）
```

**收敛性**: 通常2-3次迭代即可收敛

### 4. 装甲板匹配

**问题**: 多装甲板目标中，匹配当前检测到的装甲板

**算法**:
1. 计算EKF预测的所有装甲板位置
2. 按距离排序，取前3个
3. 计算角度误差：`|yaw_diff| + |pitch_diff|`
4. 选择角度误差最小的装甲板

**特殊处理**:
- 检测装甲板跳变（jumped）
- 记录last_id防止频繁切换

### 5. 状态机设计

**状态转换图**:
```
lost → detecting → tracking → temp_lost → lost
                ↑              ↓
                └──────────────┘
                
tracking → switching → detecting → tracking
```

**设计思路**:
- `detecting`: 需要连续检测N帧才进入tracking，防止误检
- `temp_lost`: 允许短暂丢失，提高鲁棒性
- `switching`: 全向感知场景下的目标切换

---

## 在项目中的作用

### 1. 核心功能模块

`auto_aim` 模块是整个机器人视觉系统的核心，负责：

- **感知**: 从图像中检测和识别敌方装甲板
- **决策**: 选择最优目标并跟踪
- **控制**: 计算瞄准角度并发送控制命令

### 2. 与其他模块的集成

**与IO模块**:
- `io::Camera`: 获取图像
- `io::CBoard`: 获取IMU数据和发送控制命令

**与其他任务模块**:
- `auto_buff`: 能量机关任务（共享部分基础设施）
- `omniperception`: 全向感知（提供目标切换支持）

**与工具模块**:
- `tools::Trajectory`: 弹道计算
- `tools::ExtendedKalmanFilter`: EKF实现
- `tools::math_tools`: 数学工具函数

### 3. 性能优化

- **多线程支持**: `multithread/` 目录提供多线程检测
- **模型加速**: 支持OpenVINO推理加速
- **ROI检测**: 在ROI内重新检测，提高精度

### 4. 可扩展性

- **多检测器支持**: 传统CV和YOLO可切换
- **多YOLO版本**: 支持YOLOv5/v8/11
- **配置驱动**: 所有参数通过YAML配置

---

## 使用示例

### 基本使用

```cpp
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/shooter.hpp"

// 初始化
auto_aim::Detector detector(config_path);
auto_aim::Solver solver(config_path);
auto_aim::Tracker tracker(config_path, solver);
auto_aim::Aimer aimer(config_path);
auto_aim::Shooter shooter(config_path);

// 主循环
while (running) {
    // 读取图像和IMU
    camera.read(img, timestamp);
    q = cboard.imu_at(timestamp);
    
    // 更新坐标系
    solver.set_R_gimbal2world(q);
    
    // 检测
    auto armors = detector.detect(img);
    
    // 跟踪
    auto targets = tracker.track(armors, timestamp);
    
    // 瞄准
    auto command = aimer.aim(targets, timestamp, bullet_speed);
    
    // 射击决策
    command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);
    
    // 发送命令
    cboard.send(command);
}
```

### 使用YOLO检测器

```cpp
auto_aim::YOLO yolo(config_path, false);
auto armors = yolo.detect(img);
```

### 多线程模式

```cpp
auto_aim::multithread::MultiThreadDetector detector(config_path);
auto_aim::multithread::CommandGener commandgener(shooter, aimer, cboard, plotter);

// 检测线程
std::thread detect_thread([&]() {
    while (running) {
        camera.read(img, t);
        detector.push(img, t);
    }
});

// 主线程处理检测结果
```

---

## 配置参数说明

### detector参数
- `threshold`: 二值化阈值
- `max_angle_error`: 灯条角度误差（度）
- `min_lightbar_ratio`, `max_lightbar_ratio`: 灯条长宽比范围
- `min_armor_ratio`, `max_armor_ratio`: 装甲板宽高比范围
- `min_confidence`: 分类器置信度阈值

### tracker参数
- `enemy_color`: 敌方颜色（"red"或"blue"）
- `min_detect_count`: 进入tracking需要的连续检测帧数
- `max_temp_lost_count`: 临时丢失最大帧数
- `outpost_max_temp_lost_count`: 前哨站临时丢失最大帧数

### aimer参数
- `yaw_offset`, `pitch_offset`: 角度补偿（度）
- `comming_angle`, `leaving_angle`: 小陀螺模式角度阈值（度）
- `high_speed_delay_time`, `low_speed_delay_time`: 延迟时间（秒）
- `decision_speed`: 判断高速/低速的角速度阈值（rad/s）

### shooter参数
- `first_tolerance`, `second_tolerance`: 角度容差（度）
- `judge_distance`: 判断使用哪个容差的距离阈值（米）
- `auto_fire`: 是否自动射击

---

## 调试与优化

### 调试功能

1. **可视化显示**
   - `Detector::show_result()`: 显示检测结果
   - `tools::Plotter`: 绘制目标轨迹

2. **日志输出**
   - 使用 `tools::logger()` 输出调试信息
   - 关键事件：目标切换、发散检测、收敛判断

3. **数据保存**
   - `Detector::save()`: 保存不确定的装甲板图案
   - `tools::Recorder`: 录制图像和IMU数据

### 性能优化建议

1. **检测优化**
   - 使用YOLO替代传统CV方法（更鲁棒）
   - 使用OpenVINO加速推理
   - 多线程检测

2. **跟踪优化**
   - 调整EKF参数（Q、R矩阵）
   - 优化状态机参数（temp_lost_count等）

3. **瞄准优化**
   - 调整角度补偿
   - 优化弹道迭代次数
   - 调整延迟时间

---

## 常见问题

### 1. 检测不稳定

**原因**:
- 光照变化
- 阈值设置不当
- 分类器置信度低

**解决**:
- 调整二值化阈值
- 改善光照条件
- 重新训练分类器

### 2. 跟踪丢失

**原因**:
- 目标被遮挡
- 检测失败
- EKF发散

**解决**:
- 增加temp_lost_count
- 检查检测参数
- 调整EKF初始化参数

### 3. 瞄准不准确

**原因**:
- 坐标变换错误
- 弹道计算错误
- 角度补偿不当

**解决**:
- 检查标定参数
- 验证弹道模型
- 调整角度补偿

---

## 参考资料

- OpenCV文档: https://docs.opencv.org/
- 扩展卡尔曼滤波: https://en.wikipedia.org/wiki/Extended_Kalman_filter
- PnP算法: https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
- YOLO: https://github.com/ultralytics/ultralytics

---

**最后更新**: 2025年

