# Calibration 目录说明文档

本目录包含机器人视觉系统中用于相机和手眼标定的所有工具程序。本文档详细说明各个文件的作用、使用方法和它们之间的关系。

## 📁 文件列表

1. **capture.cpp** - 数据采集工具
2. **calibrate_camera.cpp** - 相机内参标定
3. **calibrate_handeye.cpp** - 手眼标定（固定标定板）
4. **calibrate_robotworld_handeye.cpp** - 机器人-世界手眼标定（移动标定板）
5. **split_video.cpp** - 视频分割工具

---

## 🔄 标定流程概述

```
数据采集 (capture.cpp)
    ↓
    [生成: 图片 + 四元数数据]
    ↓
相机内参标定 (calibrate_camera.cpp)  ← 可选，如果已有内参可跳过
    ↓
    [生成: camera_matrix, distort_coeffs]
    ↓
手眼标定 (calibrate_handeye.cpp 或 calibrate_robotworld_handeye.cpp)
    ↓
    [生成: R_camera2gimbal, t_camera2gimbal]
```

---

## 📝 各文件详细说明

### 1. capture.cpp - 数据采集工具

#### 功能
采集标定所需的原始数据：
- **相机图像**：包含标定板的图像
- **IMU四元数**：与图像时间戳对应的云台姿态数据

#### 工作原理
1. 实时读取相机图像和IMU数据
2. 在图像上显示IMU的欧拉角（用于判断坐标系方向和零漂）
3. 检测标定板（默认10列7行圆点阵列）
4. 用户按 `s` 键保存当前帧图像和对应的四元数
5. 按 `q` 键退出程序

#### 使用方法
```bash
./capture --config-path=configs/calibration.yaml --output-folder=assets/img_with_q
```

#### 输出文件格式
- `{output_folder}/1.jpg`, `2.jpg`, ... - 图像文件
- `{output_folder}/1.txt`, `2.txt`, ... - 四元数文件（格式：w x y z）

#### 关键特性
- **时间同步**：通过时间戳确保图像和IMU数据一一对应
- **实时预览**：显示标定板识别结果和云台姿态，便于判断数据质量
- **手动筛选**：用户可以选择保存哪些帧，确保数据质量

---

### 2. calibrate_camera.cpp - 相机内参标定

#### 功能
计算相机的内参矩阵（camera_matrix）和畸变系数（distort_coeffs）

#### 工作原理
1. 读取采集的标定板图像（**只需要图片，不需要四元数**）
2. 识别每张图像中的标定板角点/圆点
3. 使用OpenCV的 `cv::calibrateCamera` 进行相机标定
4. 计算重投影误差，评估标定质量
5. 输出YAML格式的内参结果

#### 使用方法
```bash
./calibrate_camera --config-path=configs/calibration.yaml assets/img_with_q
```

#### 输入要求
- 标定板图像文件夹（包含 `1.jpg`, `2.jpg`, ...）
- 配置文件中的标定板参数：
  - `pattern_cols`: 标定板列数
  - `pattern_rows`: 标定板行数
  - `center_distance_mm`: 标定板点距（毫米）

#### 输出内容
```yaml
# 重投影误差: {error}px
camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
distort_coeffs: [k1, k2, p1, p2, k3]
```

#### 标定板类型支持

程序会自动尝试两种标定板类型：
1. **圆点阵**（优先尝试）：`cv::findCirclesGrid()`
   - 优点：对光照变化不敏感，适合工业场景
   - 缺点：检测精度相对较低（~0.5像素）
   
2. **棋盘格**（圆点阵失败后尝试）：`cv::findChessboardCorners()`
   - 优点：检测精度高（~0.1像素），经过亚像素优化后精度更高
   - 缺点：需要良好对比度
   - **自动进行亚像素优化**：`cornerSubPix()` 将角点精度优化到亚像素级

#### 注意事项
- 需要**至少10-15张**不同角度和位置的标定板图像（推荐15-20张）
- 标定板应覆盖图像的不同区域（左上、右上、左下、右下、中心）
- **支持圆点阵列和棋盘格两种标定板**，程序会自动检测
- 标定精度：棋盘格经过亚像素优化后精度更高（推荐使用棋盘格进行高精度标定）

---

### 3. calibrate_handeye.cpp - 手眼标定（固定标定板）

#### 功能
计算相机坐标系相对于云台坐标系的变换关系：**R_camera2gimbal** 和 **t_camera2gimbal**

#### 应用场景
- 标定板固定在世界坐标系中（不移动）
- 云台转动到不同姿态
- 相机安装在云台上，随云台一起运动

#### 工作原理
1. 读取图像和对应的四元数数据
2. **从图像计算**：通过 `cv::solvePnP` 得到标定板在相机坐标系下的位姿 `R_board2camera`
3. **从IMU计算**：通过四元数和 `R_gimbal2imubody` 得到云台在世界坐标系下的旋转 `R_gimbal2world`
4. 建立手眼标定方程：对于每一对数据，有：
   ```
   R_gimbal2world_i * R_camera2gimbal = R_board2world * R_camera2board_i
   ```
5. 使用 `cv::calibrateHandEye` 求解 `R_camera2gimbal` 和 `t_camera2gimbal`

#### 使用方法
```bash
./calibrate_handeye --config-path=configs/calibration.yaml assets/img_with_q
```

#### 输入要求
- 图像文件夹（包含 `.jpg` 和对应的 `.txt` 文件）
- 配置文件中的参数：
  - `camera_matrix`: 相机内参矩阵（需要先用 `calibrate_camera.cpp` 标定）
  - `distort_coeffs`: 畸变系数
  - `R_gimbal2imubody`: 云台坐标系到IMU本体坐标系的旋转矩阵（3x3，行优先）

#### 关于 R_gimbal2imubody 参数

**参数含义**：
- 表示云台坐标系到IMU本体坐标系的旋转关系
- 用于将IMU的四元数转换为云台的姿态

**默认值**：
```yaml
R_gimbal2imubody: [1, 0, 0, 0, 1, 0, 0, 0, 1]  # 单位矩阵
```
这表示IMU和云台的坐标系完全对齐（无旋转）。

**何时需要修改**：
- 当IMU安装方向与云台坐标系不一致时
- 例如：IMU的X轴对应云台的Y轴，需要绕Z轴旋转90度

**如何获取正确的值**：
1. **方法1（推荐）**：通过标定程序验证
   - 运行标定程序，观察显示的yaw/pitch/roll角度
   - 手动转动云台到已知角度（例如：yaw=45°）
   - 如果显示的角度不一致，说明`R_gimbal2imubody`需要调整
   - 根据角度差异计算正确的旋转矩阵

2. **方法2**：直接测量安装角度
   - 如果知道IMU相对云台的旋转角度（例如：绕Z轴旋转90度）
   - 使用旋转矩阵公式计算（见下方"旋转矩阵计算方法"章节）

3. **方法3**：使用Python快速计算
   ```python
   from scipy.spatial.transform import Rotation as R
   # 例如：绕Z轴旋转90度
   rot = R.from_euler('z', 90, degrees=True)
   R_matrix = rot.as_matrix()
   R_row_major = R_matrix.flatten().tolist()
   print("R_gimbal2imubody:", R_row_major)
   ```

**常见配置示例**：
- 单位矩阵（完全对齐）：`[1, 0, 0, 0, 1, 0, 0, 0, 1]`
- 绕Z轴旋转90度：`[0, 1, 0, -1, 0, 0, 0, 0, 1]`
- 绕Y轴旋转180度：`[-1, 0, 0, 0, 1, 0, 0, 0, -1]`
- 绕X轴旋转180度：`[1, 0, 0, 0, -1, 0, 0, 0, -1]`

**验证方法**：
- 标定程序会在图像上显示计算得到的云台欧拉角（yaw/pitch/roll）
- 对比显示的角度和实际云台姿态是否一致
- 如果不一致，调整`R_gimbal2imubody`直到一致

#### 输出内容
```yaml
R_gimbal2imubody: [旋转矩阵数据]
# 相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree
R_camera2gimbal: [3x3旋转矩阵]
t_camera2gimbal: [3x1平移向量，单位：米]
```

#### 关键转换链
```
IMU四元数 (imuabs系)
    ↓ (通过 R_gimbal2imubody)
云台姿态 (gimbal系 → world系)
    ↓ (通过手眼标定)
相机姿态 (camera系 → gimbal系)
```

---

### 4. calibrate_robotworld_handeye.cpp - 机器人-世界手眼标定

#### 功能
同时计算两个变换关系：
1. **R_camera2gimbal** 和 **t_camera2gimbal**（相机 → 云台）
2. **R_board2world** 和 **t_board2world**（标定板 → 世界）

#### 应用场景
- 标定板可以移动（不需要固定）
- 云台也可以转动
- 同时标定相机相对于云台的位姿和标定板在世界坐标系中的位姿

#### 工作原理
与 `calibrate_handeye.cpp` 类似，但使用 `cv::calibrateRobotWorldHandEye`，可以同时求解：
- 相机到云台的变换
- 标定板到世界的变换

#### 使用方法
```bash
./calibrate_robotworld_handeye --config-path=configs/calibration.yaml assets/img_with_q
```

#### 输出内容
```yaml
R_gimbal2imubody: [旋转矩阵数据]
# 相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree
# 标定板到世界坐标系原点的水平距离: {:.2f} m
# 标定板同竖直摆放时的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree
R_camera2gimbal: [3x3旋转矩阵]
t_camera2gimbal: [3x1平移向量，单位：米]
```

#### 与 calibrate_handeye.cpp 的区别
| 特性 | calibrate_handeye | calibrate_robotworld_handeye |
|------|-------------------|------------------------------|
| 标定板位置 | 固定在世界坐标系 | 可以移动 |
| 求解内容 | 仅相机→云台 | 相机→云台 + 标定板→世界 |
| 适用场景 | 标定板位置已知或固定 | 标定板位置未知或可变 |

#### 标定板3D点生成
注意：此程序使用的标定板3D坐标原点在标定板中心，坐标轴定义也不同：
```cpp
// 坐标系原点在标定板中心
float x = 0;
float y = (-j + 0.5 * pattern_size.width) * center_distance;
float z = (-i + 0.5 * pattern_size.height) * center_distance;
```

---

### 5. split_video.cpp - 视频分割工具

#### 功能
从录制好的视频文件中提取指定帧范围的片段，同时同步提取对应的时间戳和四元数数据

#### 工作原理
1. 读取输入视频文件（`.avi`）和对应的数据文件（`.txt`）
2. 同步提取指定帧范围的图像和数据
3. 输出新的视频文件和数据文件

#### 使用方法
```bash
./split_video input_path --start-index=100 --end-index=500 --output-path=output_path
```

#### 输入文件格式
- `{input_path}.avi` - 视频文件
- `{input_path}.txt` - 数据文件（每行格式：时间戳 w x y z）

#### 输出文件
- `{output_path}.avi` - 分割后的视频
- `{output_path}.txt` - 分割后的数据文件

#### 应用场景
- 从长时间录制的视频中提取有用的标定数据
- 数据预处理，去除无效帧

---

## 🔗 文件之间的数据流关系

### 数据依赖图

```
配置文件 (configs/calibration.yaml)
    ↓
capture.cpp ──→ [图片 + 四元数] ──→ calibrate_camera.cpp ──→ [camera_matrix, distort_coeffs]
    ↓                                                                    ↓
    └───────────────────────────────────────────────────────────────────┘
                                                                    ↓
                                                    calibrate_handeye.cpp
                                                        或
                                          calibrate_robotworld_handeye.cpp
                                                                    ↓
                                            [R_camera2gimbal, t_camera2gimbal]
```

### 配置参数依赖

```
calibration.yaml
├── pattern_cols, pattern_rows, center_distance_mm  ← capture.cpp, calibrate_camera.cpp, calibrate_handeye.cpp
├── camera_matrix, distort_coeffs                    ← calibrate_handeye.cpp, calibrate_robotworld_handeye.cpp
├── R_gimbal2imubody                                 ← calibrate_handeye.cpp, calibrate_robotworld_handeye.cpp
└── camera_name, exposure_ms, gain, CAN参数           ← capture.cpp
```

---

## 🎯 完整标定流程示例

### 步骤1：数据采集
```bash
# 运行数据采集程序
./capture --config-path=configs/calibration.yaml --output-folder=assets/img_with_q

# 操作：
# 1. 将标定板放在相机视野中
# 2. 转动云台到不同角度（至少10-15个不同姿态）
# 3. 对于每个姿态，确保标定板清晰可见，按 's' 保存
# 4. 采集足够的样本后，按 'q' 退出
```

### 步骤2：相机内参标定（如果还没有内参）
```bash
# 使用采集的图像进行相机内参标定
./calibrate_camera --config-path=configs/calibration.yaml assets/img_with_q

# 将输出的 camera_matrix 和 distort_coeffs 添加到 calibration.yaml 中
```

### 步骤3：手眼标定
```bash
# 根据实际情况选择标定方法

# 方法A：固定标定板（推荐）
./calibrate_handeye --config-path=configs/calibration.yaml assets/img_with_q

# 方法B：标定板可移动
./calibrate_robotworld_handeye --config-path=configs/calibration.yaml assets/img_with_q

# 将输出的 R_camera2gimbal 和 t_camera2gimbal 添加到相应的配置文件中
```

---

## 📊 标定质量评估

### 相机内参标定质量指标
- **重投影误差**：应该 < 0.5 像素
- 标定板图像数量：至少 15-20 张
- 标定板覆盖范围：应覆盖图像的所有区域

### 手眼标定质量检查
1. **视觉检查**：在标定程序中显示的云台欧拉角应该符合实际云台姿态
2. **转换验证**：验证 `R_gimbal2imubody` 是否正确（通过显示的yaw/pitch/roll判断）
3. **偏角检查**：输出的"相机同理想情况的偏角"应该在合理范围内（通常 < 10°）

---

## ⚠️ 注意事项

### 数据采集时
1. **时间同步**：确保相机和IMU数据的时间戳准确同步
2. **标定板质量**：标定板应该平整，图案清晰
3. **光照条件**：保持光照均匀，避免强烈反光
4. **姿态多样性**：云台应该转动到各种不同角度，包括：
   - 不同的yaw角度
   - 不同的pitch角度
   - 标定板在图像的不同位置

### 标定板要求
- **圆点阵列**：默认10列7行，点距40mm
- **棋盘格**：也可以使用棋盘格（在 `calibrate_camera.cpp` 中支持）
- **尺寸精度**：标定板的实际尺寸必须与配置文件中一致

### 坐标系定义
- **IMU坐标系（imuabs）**：绝对坐标系，由IMU定义
- **IMU本体坐标系（imubody）**：IMU的本体坐标系
- **云台坐标系（gimbal）**：云台的坐标系，通过 `R_gimbal2imubody` 与IMU关联
- **相机坐标系（camera）**：相机的坐标系
- **世界坐标系（world）**：固定的世界坐标系

### 关于IMU与云台的平移偏移

**当前实现**：
- 代码中只使用旋转矩阵 `R_gimbal2imubody`，**不需要平移向量**
- 假设云台在世界坐标系原点（`t_gimbal2world = 0`）

**原因说明**：
1. **手眼标定主要求解旋转关系**：`R_camera2gimbal` 的求解不依赖平移偏移
2. **IMU只提供旋转信息**：四元数只包含姿态，不包含位置信息
3. **平移偏移的影响**：
   - 如果IMU不在云台转轴上，云台旋转时IMU位置会改变
   - 但这不影响旋转关系（`R_gimbal2imubody`）的求解
   - 只影响云台的绝对位置（如果需要精确位置，可能需要额外考虑）

**结论**：
- **对于手眼标定**：只需要 `R_gimbal2imubody`，不需要平移向量
- **如果IMU与云台转轴距离 < 5cm**：平移偏移影响可忽略
- **如果需要云台的精确绝对位置**：可能需要额外的定位系统（当前代码未实现）

---

## 🛠️ 故障排除

### 问题1：标定板识别失败
- **原因**：光照不足、标定板图案不清晰、标定板尺寸不匹配
- **解决**：
  - 改善光照条件（均匀照明，避免强反光）
  - 检查标定板尺寸配置（`pattern_cols`, `pattern_rows`, `center_distance_mm`）
  - 确保标定板平整（不能有褶皱或弯曲）
  - 检查标定板是否完全在视野内
  - 如果圆点阵识别失败，程序会自动尝试棋盘格

### 问题2：手眼标定结果不准确
- **原因**：
  - `R_gimbal2imubody` 配置错误（最常见）
  - 数据采集角度不够多样
  - 标定板位置固定不好
  - 数据样本数量不足（< 10组）
- **解决**：
  - **验证 `R_gimbal2imubody`**：
    1. 运行标定程序，观察显示的yaw/pitch/roll角度
    2. 手动转动云台到已知角度（例如：yaw=0°, pitch=30°）
    3. 如果显示角度与实际不一致，说明`R_gimbal2imubody`错误
    4. 根据角度差异计算正确的旋转矩阵
  - **增加数据多样性**：至少采集15-20组不同姿态的数据
  - **确保标定板固定**：如果使用`calibrate_handeye.cpp`，标定板必须固定
  - **检查重投影误差**：相机内参标定的重投影误差应 < 0.5像素

### 问题3：四元数数据无法读取
- **原因**：CAN通信故障、IMU未正确初始化、文件不存在
- **解决**：
  - 检查CAN接口配置（`can_interface: "can0"`）
  - 确认CAN总线正常工作：`ip link show can0`
  - 检查IMU设备是否在线
  - 确认`.txt`文件存在且格式正确（w x y z，空格分隔）

### 问题4：相机内参标定重投影误差过大（> 0.5像素）
- **原因**：
  - 标定板图像质量差
  - 标定板姿态角度不够多样
  - 图像数量不足
  - 标定板尺寸配置错误
- **解决**：
  - 使用棋盘格并确保亚像素优化成功
  - 增加标定板图像数量（至少15-20张）
  - 确保标定板覆盖图像所有区域
  - 检查`center_distance_mm`是否正确（单位：毫米）

### 问题5：显示的云台欧拉角与实际不符
- **原因**：`R_gimbal2imubody` 配置错误
- **解决**：
  1. 记录显示的欧拉角和实际云台姿态
  2. 计算角度差异
  3. 根据差异计算正确的旋转矩阵（见"旋转矩阵计算方法"章节）
  4. 更新配置文件中的`R_gimbal2imubody`
  5. 重新运行标定程序验证

---

## 🔧 旋转矩阵计算方法

### 绕坐标轴旋转的基本公式

#### 绕X轴旋转角度θ
```
[1,     0,        0   ]
[0,  cos(θ), -sin(θ) ]
[0,  sin(θ),  cos(θ) ]
```

#### 绕Y轴旋转角度θ
```
[ cos(θ),  0,  sin(θ) ]
[    0,    1,     0   ]
[-sin(θ),  0,  cos(θ) ]
```

#### 绕Z轴旋转角度θ
```
[cos(θ), -sin(θ),  0]
[sin(θ),  cos(θ),  0]
[  0,       0,     1]
```

### 计算示例：绕Y轴旋转180度

**步骤1**：代入角度
```
θ = 180°
cos(180°) = -1
sin(180°) =  0
```

**步骤2**：代入公式
```
[ cos(180°),  0,  sin(180°) ]   [ -1,  0,  0 ]
[    0,       1,     0      ] = [  0,  1,  0 ]
[-sin(180°),  0,  cos(180°) ]   [  0,  0, -1 ]
```

**步骤3**：转换为行优先格式（用于YAML）
```
矩阵格式：
[-1,  0,  0]  ← 第一行
[ 0,  1,  0]  ← 第二行
[ 0,  0, -1]  ← 第三行

行优先格式（按行展开）：
[-1, 0, 0, 0, 1, 0, 0, 0, -1]
```

**YAML配置**：
```yaml
R_gimbal2imubody: [-1, 0, 0, 0, 1, 0, 0, 0, -1]
```

### 使用Python快速计算

```python
from scipy.spatial.transform import Rotation as R

# 方法1：绕单个轴旋转
rot = R.from_euler('y', 180, degrees=True)  # 绕Y轴旋转180度
R_matrix = rot.as_matrix()
R_row_major = R_matrix.flatten().tolist()
print("R_gimbal2imubody:", R_row_major)

# 方法2：组合旋转（例如：先绕Z轴90度，再绕Y轴180度）
rot = R.from_euler('zyx', [90, 180, 0], degrees=True)  # 欧拉角顺序：ZYX
R_matrix = rot.as_matrix()
R_row_major = R_matrix.flatten().tolist()
print("R_gimbal2imubody:", R_row_major)
```

### 常见旋转配置

| 旋转情况 | 旋转矩阵（行优先） | 说明 |
|---------|------------------|------|
| **无旋转（对齐）** | `[1, 0, 0, 0, 1, 0, 0, 0, 1]` | 单位矩阵，默认配置 |
| **绕Z轴旋转90°** | `[0, -1, 0, 1, 0, 0, 0, 0, 1]` | X→Y, Y→-X |
| **绕Z轴旋转-90°** | `[0, 1, 0, -1, 0, 0, 0, 0, 1]` | X→-Y, Y→X |
| **绕Y轴旋转180°** | `[-1, 0, 0, 0, 1, 0, 0, 0, -1]` | X→-X, Z→-Z |
| **绕X轴旋转180°** | `[1, 0, 0, 0, -1, 0, 0, 0, -1]` | Y→-Y, Z→-Z |

---

## 📐 相机参数与标定板大小的关系

### 焦距、视场角与标定板大小

#### 基本关系
```
视场角 FOV = 2 × arctan(传感器尺寸 / (2 × 焦距))
标定板在图像中的大小 ∝ 物理尺寸 × 焦距 / 距离
```

#### 标定板大小建议
- **标定板应覆盖视野的 25%-50%**
  - 太小：检测困难，精度低
  - 太大：无法完整看到，标定失败

#### 标定距离计算
```
推荐距离 ≈ 标定板物理尺寸 / (tan(FOV/2) × 期望覆盖比例)
```

#### 实际配置示例
对于 **360mm × 240mm** 标定板（10列×7行，40mm点距）：

| 相机FOV | 推荐标定距离 | 标定板占视野比例 |
|---------|-------------|----------------|
| **87.7°** (广角) | 0.5 - 1.5m | 30-50% ✓ |
| **50°** (中焦) | 0.7 - 1.2m | 35-50% ✓ |
| **30°** (长焦) | 1.0 - 2.0m | 40-60% ✓ |

#### 注意事项
- **焦距影响**：焦距越大，视野越小，标定板需要更远
- **标定板尺寸**：当前配置（360mm×240mm）适合大多数工业相机
- **点距要求**：角点/圆点间距在图像中应至少20-30像素

---

## 📋 完整参数列表

### 执行完整标定流程后获得的参数

#### 1. 相机内参（来自 `calibrate_camera.cpp`）
```yaml
camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]  # 9个元素
distort_coeffs: [k1, k2, p1, p2, k3]             # 5个元素
# 重投影误差: {error}px                           # 质量指标
```

#### 2. 手眼标定参数（来自 `calibrate_handeye.cpp` 或 `calibrate_robotworld_handeye.cpp`）
```yaml
R_gimbal2imubody: [r11, r12, r13, r21, r22, r23, r31, r32, r33]  # 9个元素
R_camera2gimbal: [r11, r12, r13, r21, r22, r23, r31, r32, r33]  # 9个元素
t_camera2gimbal: [tx, ty, tz]                                   # 3个元素，单位：米
# 相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree  # 诊断信息
```

#### 3. 额外信息（仅 `calibrate_robotworld_handeye.cpp`）
```yaml
# 标定板到世界坐标系原点的水平距离: {:.2f} m
# 标定板同竖直摆放时的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree
```

**参数总计**：约 **26-29个核心参数**（取决于标定方法）

---

## 📚 相关资源

- OpenCV 手眼标定文档：https://docs.opencv.org/
- 机器人手眼标定原理：需要理解坐标系变换和手眼标定方程
- 四元数基础：理解四元数在姿态表示中的应用

---

## 📝 更新日志

- **2025年**：本文档创建
- **包含的工具版本**：基于OpenCV的手眼标定实现
- **主要更新**：
  - 添加了标定板类型自动检测说明
  - 补充了`R_gimbal2imubody`参数的详细计算方法
  - 添加了旋转矩阵计算方法和常见配置示例
  - 增加了相机参数与标定板大小的关系说明
  - 扩展了故障排除章节

---

## 💡 实用提示

1. **标定板选择**：
   - 如果追求高精度，推荐使用**棋盘格**（经过亚像素优化后精度可达0.1像素）
   - 如果环境光照不稳定，使用**圆点阵**（对光照变化不敏感）

2. **`R_gimbal2imubody` 配置**：
   - **推荐方法**：先用单位矩阵，运行标定程序观察显示的欧拉角
   - 如果与实际云台姿态不一致，根据角度差异计算正确的旋转矩阵
   - 使用Python脚本可以快速计算旋转矩阵

3. **数据采集技巧**：
   - 采集时确保标定板清晰可见（不要模糊）
   - 尽量覆盖云台的各种姿态角度
   - 建议采集15-20组数据（不要少于10组）

4. **标定质量检查**：
   - 重投影误差应 < 0.5像素
   - 显示的云台欧拉角应与实际姿态一致
   - 相机偏角应在合理范围内（< 10°）

---

**提示**：标定是一个需要耐心和细致的过程，建议在标定前充分理解各个坐标系之间的关系，确保配置参数正确，并采集足够多样化的数据样本。
