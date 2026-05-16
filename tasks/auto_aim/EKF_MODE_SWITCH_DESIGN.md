# EKF 横移 / 自转模式切换权重设计

## 1. 背景问题

当前自瞄跟踪器使用 `Target` 内部的 EKF 估计目标状态，状态向量定义在
`tasks/auto_aim/target.cpp`：

```text
x vx y vy z vz a w r l h
```

其中：

```text
x/y/z   : 机器人旋转中心位置
vx/vy/vz: 机器人旋转中心速度
a       : 目标当前旋转角
w       : 目标角速度
r       : 主半径，即旋转中心到一组装甲板中心的水平距离
l       : 长短半径差，另一组装甲板半径约为 r + l
h       : 高低装甲板的高度差
```

原始 EKF 在对方机器人快速自转时收敛较快，因为 `angle` 观测权重较高，角度残差能快速修正 `a/w`。

但当对方机器人快速左右横移时，`angle` 残差中会混入横移、PnP 抖动、视角变化导致的误差。此时如果 EKF 仍然高度相信 `angle`，滤波器可能会用修改 `r/l/h` 的方式去解释这些残差，导致半径变大，进而让预测装甲板被重投影得偏近、偏大。

因此需要做“横移 / 自转模式切换”的观测权重策略：

```text
自转主导：相信 angle，用于快速收敛 a/w
横移主导：降低 angle 对结构参数的影响，保护 r/l/h
```

## 2. 核心目标

本设计不是简单把 `angle` 权重永久调小，而是根据运动状态动态切换权重。

目标如下：

1. 目标真实自转时，`a/w` 仍然快速收敛。
2. 目标快速横移时，`r/l/h` 不被横移误差拉飞。
3. 检测装甲板和 EKF 预测装甲板的重投影框尺寸保持接近。
4. 模式切换应平滑，避免一帧抖动导致权重突然跳变。

## 3. 可用观测量

当前代码中可直接使用的数据包括：

```text
armor.xyz_in_world      : 当前检测装甲板在世界坐标系下的位置
armor.ypd_in_world      : 当前检测装甲板的 yaw / pitch / distance
armor.ypr_in_world      : 当前检测装甲板的 yaw / pitch / roll 姿态
ekf_.x[1], ekf_.x[3]    : 目标中心在 x/y 方向的速度估计
ekf_.x[7]               : 目标角速度 w
is_switch_              : 是否发生装甲板 id 切换
delta_angle             : 当前装甲板朝向与中心视线夹角
```

建议派生两个指标：

```text
linear_speed = sqrt(vx^2 + vy^2)
angular_speed = abs(w)
```

还可以使用一个横移视角指标：

```text
abs_delta_angle = abs(limit_rad(armor.ypr_in_world[0] - atan2(armor.xyz_in_world[1], armor.xyz_in_world[0])))
```

`abs_delta_angle` 越大，说明装甲板朝向和相机视线越不一致，PnP 姿态和 `angle` 观测越容易不稳定。

## 4. 模式定义

建议先定义三个模式，而不是只有横移 / 自转两个模式。

```text
ROTATION_DOMINANT   : 自转主导
TRANSLATION_DOMINANT: 横移主导
MIXED               : 混合或不确定
```

判断思路：

```text
如果 angular_speed 明显大，且 linear_speed 不大 -> ROTATION_DOMINANT
如果 linear_speed 明显大，且 angular_speed 不大 -> TRANSLATION_DOMINANT
否则 -> MIXED
```

建议初始阈值：

```text
linear_speed_high  = 1.2   m/s
linear_speed_low   = 0.6   m/s
angular_speed_high = 1.5   rad/s
angular_speed_low  = 0.8   rad/s
delta_angle_high   = 0.45  rad
```

可以用迟滞避免模式来回跳：

```text
进入横移模式：linear_speed > linear_speed_high 或 abs_delta_angle > delta_angle_high
退出横移模式：linear_speed < linear_speed_low 且 abs_delta_angle < 0.30

进入自转模式：angular_speed > angular_speed_high 且 linear_speed < linear_speed_high
退出自转模式：angular_speed < angular_speed_low
```

## 5. 权重策略

EKF 更新时的观测噪声矩阵 `R` 顺序是：

```text
yaw, pitch, distance, angle
```

`R` 越小，表示越相信该观测；`R` 越大，表示越不相信该观测。

### 5.1 自转主导

自转主导时，`angle` 是有价值的真实旋转信息，应提高其权重。

建议：

```text
yaw_var      = 4e-4
pitch_var    = 4e-4
distance_var = pow(0.03 + 0.015 * distance, 2)
angle_var    = pow(0.06 + 0.10 * abs_delta_angle, 2)
structure_alpha = 0.05
```

解释：

```text
angle_var 较小，让 a/w 快速收敛。
structure_alpha 很小，让 r/l/h 慢更新，避免自转时结构参数被错误修改。
```

### 5.2 横移主导

横移主导时，`angle` 中容易混入横移和 PnP 抖动，不应强力驱动 EKF。

建议：

```text
yaw_var      = 4e-4
pitch_var    = 4e-4
distance_var = pow(0.02 + 0.01 * distance, 2)
angle_var    = pow(0.18 + 0.45 * abs_delta_angle, 2)
structure_alpha = 0.02
```

解释：

```text
distance_var 较小，保留距离约束，防止预测框变大。
angle_var 较大，降低横移误差对 EKF 的影响。
structure_alpha 很小，基本冻结 r/l/h。
```

### 5.3 混合模式

混合模式适合普通运动，不激进相信任何单一路径。

建议：

```text
yaw_var      = 4e-4
pitch_var    = 4e-4
distance_var = pow(0.02 + 0.01 * distance, 2)
angle_var    = pow(0.12 + 0.35 * abs_delta_angle, 2)
structure_alpha = 0.10
```

## 6. 关键实现思想

重点不是让 `angle` 对所有状态都变弱，而是区分两类状态：

```text
a/w     : 运动状态，可以更快跟随 angle
r/l/h   : 结构状态，应缓慢变化，甚至在横移时冻结
```

当前 `ExtendedKalmanFilter::update()` 是整体更新状态向量，因此一个实用做法是：

1. 正常执行 EKF update。
2. update 前保存 `prev_r/l/h`。
3. update 后只对 `r/l/h` 做平滑回拉。
4. 根据模式选择不同 `structure_alpha`。

公式：

```text
r_new = r_prev + structure_alpha * (r_ekf - r_prev)
l_new = l_prev + structure_alpha * (l_ekf - l_prev)
h_new = h_prev + structure_alpha * (h_ekf - h_prev)
```

这等价于：

```text
a/w 继续吃 EKF 更新
r/l/h 只吃很小一部分结构更新
```

## 7. 伪代码

建议在 `Target::update_ypda()` 中实现。

```cpp
enum class MotionMode
{
  ROTATION_DOMINANT,
  TRANSLATION_DOMINANT,
  MIXED
};

MotionMode estimate_motion_mode(
  double linear_speed,
  double angular_speed,
  double abs_delta_angle)
{
  if ((linear_speed > 1.2 || abs_delta_angle > 0.45) && angular_speed < 1.5) {
    return MotionMode::TRANSLATION_DOMINANT;
  }

  if (angular_speed > 1.5 && linear_speed < 1.2) {
    return MotionMode::ROTATION_DOMINANT;
  }

  return MotionMode::MIXED;
}
```

权重选择：

```cpp
double yaw_var = 4e-4;
double pitch_var = 4e-4;
double distance_var = 0.0;
double angle_var = 0.0;
double structure_alpha = 0.0;

switch (mode) {
  case MotionMode::ROTATION_DOMINANT:
    distance_var = std::pow(0.03 + 0.015 * distance, 2);
    angle_var = std::pow(0.06 + 0.10 * abs_delta_angle, 2);
    structure_alpha = 0.05;
    break;

  case MotionMode::TRANSLATION_DOMINANT:
    distance_var = std::pow(0.02 + 0.01 * distance, 2);
    angle_var = std::pow(0.18 + 0.45 * abs_delta_angle, 2);
    structure_alpha = 0.02;
    break;

  case MotionMode::MIXED:
    distance_var = std::pow(0.02 + 0.01 * distance, 2);
    angle_var = std::pow(0.12 + 0.35 * abs_delta_angle, 2);
    structure_alpha = 0.10;
    break;
}
```

EKF 更新后处理：

```cpp
auto prev_r = ekf_.x[8];
auto prev_l = ekf_.x[9];
auto prev_h = ekf_.x[10];

ekf_.update(z, H, R, h, z_subtract);

ekf_.x[8] = prev_r + structure_alpha * (ekf_.x[8] - prev_r);
ekf_.x[9] = prev_l + structure_alpha * (ekf_.x[9] - prev_l);
ekf_.x[10] = prev_h + structure_alpha * (ekf_.x[10] - prev_h);
```

## 8. 为什么不直接固定半径

直接固定 `r/l/h` 最稳定，但会失去适应能力。

例如：

```text
不同兵种半径不同
PnP 初始误差可能需要一点时间修正
装甲板类型识别或初值可能不准
```

更推荐的做法是：

```text
初始化阶段允许较快收敛
收敛后结构参数慢更新
横移时几乎冻结结构参数
自转时允许 a/w 快速更新，但 r/l/h 仍慢更新
```

也就是说，结构参数不是不能变，而是不应该被短时横移残差大幅改变。

## 9. 验证指标

建议在日志中打印以下量：

```text
mode
linear_speed
angular_speed
abs_delta_angle
r = x[8]
l = x[9]
h = x[10]
residual_distance
residual_angle
```

理想现象：

```text
快速自转：
  mode 应多为 ROTATION_DOMINANT
  w 收敛快
  r/l/h 小幅变化

快速横移：
  mode 应多为 TRANSLATION_DOMINANT
  r/l/h 基本不爬升
  预测框与检测框尺寸接近

普通运动：
  mode 应多为 MIXED
  r/l/h 缓慢收敛，不跳变
```

## 10. 推荐调参顺序

1. 先固定 `structure_alpha`：

```text
TRANSLATION_DOMINANT = 0.02
ROTATION_DOMINANT    = 0.05
MIXED                = 0.10
```

2. 再调 `angle_var`：

```text
横移时 r 还变大 -> 增大 TRANSLATION_DOMINANT 的 angle_var
自转时 w 跟不上 -> 减小 ROTATION_DOMINANT 的 angle_var
```

3. 最后调模式阈值：

```text
横移识别不出来 -> 降低 linear_speed_high 或 delta_angle_high
自转识别不出来 -> 降低 angular_speed_high
模式乱跳 -> 加大 high/low 阈值差距
```

## 11. 最小落地位置

建议先只改两个文件：

```text
tasks/auto_aim/target.hpp
tasks/auto_aim/target.cpp
```

如果要做模式迟滞，需要在 `Target` 中保存上一次模式：

```cpp
MotionMode motion_mode_;
```

如果只做无迟滞版本，可以暂时只在 `update_ypda()` 内用局部变量判断模式。

## 12. 风险与注意事项

1. `linear_speed` 来自 EKF 自身，目标刚初始化时不够可靠，前几帧建议默认 `MIXED`。
2. `angular_speed` 也来自 EKF，目标切换时可能有短暂尖峰，建议配合 `is_switch_` 降低结构更新。
3. `abs_delta_angle` 依赖 PnP 姿态，侧向装甲板时可能抖动，需要和速度指标一起看。
4. 不建议让 `r/l/h` 在横移时完全不更新太久，否则初始半径错误时可能收敛慢。
5. 推荐保留物理限幅，作为最后一道保护。

## 13. 一句话总结

横移 / 自转模式切换的核心思想是：

```text
让 angle 继续帮助 a/w 跟踪真实自转，
但不要让横移造成的 angle 残差去改写 r/l/h 这种结构参数。
```

