# 配置参数说明与使用映射 (Configuration Parameters Overview)

本文件汇总 `configs/` 下各 YAML 的主要参数、含义、默认取值意图以及在代码中的典型使用位置。用于快速理解调参影响与迁移到自定义机器人时的修改入口。

> 约定：文件中未出现的少量参数（如仅某机型出现的临时试验参数）可在对应源码中进一步确认。推断来源基于命名与常见视觉/控制链路。若需要更精确定位，可在源码中 grep 关键字。

---
## 统一参数分类

| 分类 | 代表参数 | 作用概述 | 典型使用模块/文件 |
|------|----------|----------|------------------|
| 基本策略 | `enemy_color` | 指示敌方颜色（影响颜色阈值或识别过滤） | `tasks/auto_aim/detector.*`，`buff_detector.*` |
| 模型选择 | `yolo_name`, `yolo11_model_path`, `yolov5_model_path`, `yolov8_model_path`（IR模型，用于检测装甲板，支持NPU）, `classify_model`（用来分类数字）, `device`（推理设备：`CPU`, `GPU`, `"AUTO:NPU,CPU"` 等，支持自动选择设备）, `use_traditional`（判断是否使用传统方法去矫正） | 选择检测与分类模型、推理设备 (CPU/GPU/NPU) | `tasks/auto_aim/yolo.*`, `classifier.*`, YOLO 版本适配器 |
| 传统检测阈值 detect.cpp| `threshold`, `max_angle_error`, `min_lightbar_ratio`, `max_lightbar_ratio`, `min_lightbar_length`, `min_armor_ratio`, `max_armor_ratio`, `max_side_ratio`, `max_rectangular_error`, `min_confidence` | 光条与装甲板几何/强度筛选阈值 | `detector.*` (几何检查、光条拟合、角点回归) |
| ROI （用于YOLO的目标检测中）| `roi.{x,y,width,height}`, `use_roi` | 仅在检测/跟踪计算中裁剪输入区域以加速与降噪（不是控制显示），YOLO 后处理会用 offset=(x,y) 还原到原图坐标 | `tasks/auto_aim/yolos/*`（推理前裁剪，后处理加偏移） |
| 跟踪器 （按距离和装甲板优先级排序）| `min_detect_count`, `max_temp_lost_count`, `outpost_max_temp_lost_count` | 确认跟踪成立所需帧数、临时丢失容忍帧数、特殊目标(前哨站)丢失容忍 | `tracker.*`, `target.*` | （src/sentry_multithread.cpp使用了全向感知）
| 自瞄瞄准 | `yaw_offset`, `pitch_offset`（最终下发给云台的俯仰/偏航角进行固定偏置补偿，度，弥补安装误差或系统标定残差）, `comming_angle`（进入扇区角度阈值，度，用于判断目标是否进入射击范围）, `leaving_angle`（离开扇区角度阈值，度）, `decision_speed`（判断目标是"高速"旋转还是"低速"的角速度阈值，rad/s，用于选择不同的时间延迟参数）, `high_speed_delay_time`（高速目标延迟时间，秒，包含弹道飞行时间与通信延迟）, `low_speed_delay_time`（低速目标延迟时间，秒）, `left_yaw_offset`, `right_yaw_offset`（双头射击模式的左右偏置补偿，度，可选参数）, `min_spin_speed`（最小旋转速度阈值，rad/s，用于区分静止与旋转目标，无人机等场景使用） | 角度补偿、进入/离开扇区判定、转速阈值、延迟补偿（弹道+通信） | `aimer.*`, `shooter.*` |
| 射击策略 | `first_tolerance`（近距离射击允许的角度误差，度）, `second_tolerance`（远距离射击允许的角度误差，度）, `judge_distance`（用于判定目标"近/远"的距离阈值，米）, `auto_fire`（判断是否自动开火，布尔值）, `fire_gap_time`（两次开火之间的最小间隔，秒，防止连续过快开火）, `predict_time`（预测时间窗口，秒，用于 BUFF 预测旋转目标未来位置）, `aim_time`（预瞄时间，秒，BUFF 专用，部分配置中未使用）, `wait_time`（等待稳定时间，秒，BUFF 专用，部分配置中未使用） | 命中窗口宽容度、远近距离阈值、是否自动开火、BUFF 撞击预测与开火节奏 | `shooter.cpp`, `buff_aimer.*` |
| 标定与外参 | `camera_matrix`（相机内参矩阵）, `distort_coeffs`（相机畸变系数）, `R_camera2gimbal`（相机坐标系到“云台（gimbal）”坐标系的旋转矩阵）, `t_camera2gimbal`（相机坐标系到 gimbal 坐标系的平移向量）, `R_gimbal2imubody`（云台坐标系到 IMU（或机体）坐标系之间的旋转矩阵，与云台和imu的安装关系对应） | 相机内外参，坐标系转换矩阵与平移，用于像素↔世界/云台坐标转换 | `solver.*`, `trajectory.*`, PnP & 重投影误差优化 |
| 工业相机参数 | `camera_name`（相机类型：`"hikrobot"`, `"hikrobot_gige"`, `"mindvision"` 等）, `exposure_ms`（曝光时间，毫秒）, `gain`（增益，数值范围依相机而定）, `gamma`（伽马值，仅部分相机支持，如 mindvision）, `vid_pid`（USB 设备的 VID:PID，用于多相机时区分设备，格式：`"2bdf:0001"`） | 工业相机硬件采集配置 | `io/camera.*`, `io/hikrobot/*`, `io/mindvision/*` |
| USB相机参数 | `image_width`, `image_height`（采集分辨率，像素）, `usb_frame_rate`（目标帧率，FPS）, `usb_exposure`（曝光，设备单位，范围通常 1-80000）, `usb_gamma`（伽马值，通常 0-255）, `usb_gain`（增益，通常 0-96）, `new_usb_exposure`（可选，用于动态调整曝光）, `fov_h`, `fov_v`（水平/垂直视场角，度）, `new_image_width`, `new_image_height`（算法处理分辨率，像素）, `new_fov_h`, `new_fov_v`（处理后的视场角，度） | USB 相机硬件采集配置、视场角（用于视场内角度换算与 ROI 决策） | `io/usbcamera.*`, FOV 在 `decider.*` 中用于角度归一化 |
| CAN （与Cboard通信）| `quaternion_canid`（接收 IMU 四元数的 CAN ID，数据：（x,y,z,w））, `bullet_speed_canid`（接收弹速与模式信息的 CAN ID，数据包括：接收弹速、模式、射击模式、FT 角）, `send_canid`（下发控制命令的 CAN ID，数据包括：是否控制，是否发射，yaw,pitch,horizon_distance(水平距离，无人机使用)）, `can_interface`, `com_port` | 设备通信 ID 与接口名（IMU队列、弹速更新、指令下发） | `io/cboard.*`, `io/gimbal.*` |
| MPC / 规划Planner | `fire_thresh`（决策“是否开火”的阈值。代码里把 yaw 与 pitch 的误差（在预测时间点）做欧氏距离（hypot），若小于 fire_thresh 则认为命中可信可以开火。单位：弧度（rad）。原因：整个 Planner 中角度/速度均以弧度/弧度每秒为单位处理）, `max_yaw_acc`, `max_pitch_acc`, （对控制输入 u 的上下界限制（MPC 中的加速度限制）。在模型里状态是 [angle, angular_velocity]，控制量 u 对速度积分（因此 u 是加速度）。
单位：弧度/秒^2（rad/s^2））`Q_yaw`,`Q_pitch`,（状态成本矩阵 Q 的对角元素（state 维度为 2），通常形如 [w_angle, w_rate]，在 MPC 的二次目标中权衡对角/速度误差的惩罚强度，Q[0] 越大 => 更强烈地惩罚角度偏差；Q[1] 越大 => 惩罚角速度偏差（能抑制震荡或不希望的大速度）） `R_yaw`, `R_pitch`（控制输入（u）的代价，对应 R（通常是 1×1，因为只有一个输入维度），用于抑制过大的控制（加速度）输出，R 越大 => MPC 更不愿意使用大加速度（更“保守”）） | 规划器约束与代价权重，决定角速度/加速度优化与触发阈值 | `planner/planner.*`, `tinympc/*.cpp` |
| BUFF 检测显式参数 | `model`（BUFF 专用 YOLO 模型文件路径，必需）, `detect.contrast`（对比度调整系数）, `detect.brightness.{blue,red}`（红蓝方亮度偏移，通常负值降低亮度）, `detect.brightness_threshold.{blue,red}`（红蓝方二值化阈值）, `detect.morphology_size.{blue,red}`（形态学操作核大小，用于去噪）, `detect.dilate_size`（膨胀操作次数）, `R_contours_min_area`, `R_contours_max_area`（R 字母轮廓面积范围，像素²）, `fanblades_head_contours_min_area`, `fanblades_head_contours_max_area`（扇叶头部轮廓面积范围）, `fanblades_body_contours_min_area`, `fanblades_body_contours_max_area`（扇叶主体轮廓面积范围）, `standard_fanblade_path`（标准扇叶模板图片路径，用于模板匹配） | 图像预处理亮度调节、二值化阈值、形态学核大小、候选轮廓面积过滤、标准扇叶模板路径、BUFF YOLO 模型路径 | `buff_detector.*`, `yolo11_buff.*`, `img_tools.*` |
| 决策模式 | `mode` （用于选择不同的“优先级映射”策略（priority map），从而改变系统对检测到的不同装甲板类型（ArmorName）的优先级排序，最终影响目标选择和云台下发的瞄准命令）
（MODE_ONE (mode = 1) 的映射（mode1）：
three -> first
four -> first
one -> second
five -> third
sentry-> third
two -> forth
outpost -> fifth
base -> fifth
not_armor -> fifth）
MODE_TWO (mode = 2) 的映射（mode2）：
two -> first
one -> second
three -> second
four -> second
five -> second
sentry-> third
outpost -> third
base -> third
not_armor -> third| 决策优先级映射选择（不同敌方角色聚焦策略） | `omniperception/decider.*` |
| Gimbal PID | `com_port`(云台串口设备路径（例如 /dev/gimbal），用于与云台控制器通过串口协议通信),`yaw_kp`, `yaw_kd`, `pitch_kp`, `pitch_kd`(云台 PID（视觉侧）”的增益系数) | 若启用视觉侧 PID 增益（或传递给云台控制） | `gimbal.*` / 可能在高层控制环使用 |

---
## 详细参数说明按 YAML 列表

### 1. `ascento.yaml` / `demo.yaml` / `standard*.yaml` / `sentry.yaml` / `uav.yaml`
这些文件结构类似（角色不同，数值差异）：
- `enemy_color`: 目标阵营颜色，用于颜色过滤与优先级（在 BUFF / 自瞄色彩判断阶段）。
- 模型路径 (`yolo11_model_path`, `yolov5_model_path`, `yolov8_model_path`, `classify_model`): 决定加载哪种检测与分类模型，`yolo_name` 为当前使用版本标签（如 `"yolov5"`, `"yolov8"`, `"yolo11"`）。`device` 指定推理设备：
  - `"CPU"`: 使用 CPU 推理
  - `"GPU"`: 使用 GPU 推理（需要 OpenVINO GPU 插件）
  - `"AUTO:NPU,CPU"`: 自动选择设备，优先使用 NPU，失败则回退到 CPU（适用于支持 NPU 的硬件平台）
  - `"AUTO"`: OpenVINO 自动选择最优设备
- `use_traditional`: 是否启用传统几何分支与 YOLO 结果互补，提高低光或遮挡鲁棒性。
- ROI 与 `use_roi`: 开启后对输入图像裁剪提高速度；对应在检测前对 `cv::Mat` 做 ROI。
- 传统检测阈值组：控制光条形态筛选（长宽比、角度误差、矩形拟合误差）、装甲板宽高比、最小光条长度。
- 跟踪器参数：防止瞬时噪声；`outpost_max_temp_lost_count` 针对前哨站长时间遮挡。
- Aimer 参数：角度偏置（硬件安装误差补偿）、进入/离开扇区角度（用于旋转目标）、决策用目标角速度门限，延迟时间模拟弹道/处理延迟。
- Shooter 参数：不同距离射击允许误差（影响是否开火），`judge_distance` 用于区分近/远目标策略，`auto_fire` 控制是否自动触发。
- 工业相机参数 (`camera_name`, `exposure_ms`, `gain`, `gamma`, `vid_pid`): 
  - `camera_name`: 相机类型标识，如 `"hikrobot"`（海康 USB 相机）、`"hikrobot_gige"`（海康 GigE 相机）、`"mindvision"`（迈德威视相机）
  - `exposure_ms`: 曝光时间，单位毫秒（ms）
  - `gain`: 增益值，数值范围依相机型号而定（通常 0-100）
  - `gamma`: 伽马值，仅部分相机支持（如 mindvision），用于亮度曲线调整
  - `vid_pid`: USB 设备的供应商 ID:产品 ID，格式为 `"VID:PID"`（如 `"2bdf:0001"`），用于多相机时区分设备，在 `io/camera.*` 中用于设备过滤
- 标定外参组：`R_gimbal2imubody`, `R_camera2gimbal`, `t_camera2gimbal`, `camera_matrix`, `distort_coeffs` 在 `solver` 做像素→世界坐标、重投影误差优化及姿态调整。
- CAN / 串口 ID：用于 `CBoard` 接收 IMU 四元数与弹速, 以及发送控制指令 (`send_canid`)；`com_port` 串口路径用于云台通信重连。
- Planner 参数（仅出现在含 MPC 的配置如 `standard4.yaml` / `sentry.yaml`）：`fire_thresh` 触发高级控制或开火条件阈值；`Q_*`, `R_*` 为状态/测量权重，`max_*_acc` 为加速度约束。
- BUFF 检测 / 瞄准参数（在多功能角色如 `standard3.yaml` / `standard4.yaml` / `sentry.yaml` / `uav.yaml` / `mvs.yaml` 中出现）：
  - `model`: BUFF 专用 YOLO 模型路径（必需），如 `"assets/yolo11_buff_int8.xml"`
  - `detect.contrast`: 对比度调整系数（通常 0.5-2.0）
  - `detect.brightness.{blue,red}`: 红蓝方亮度偏移值（通常负值，如 -100 到 -120，用于降低亮度突出目标）
  - `detect.brightness_threshold.{blue,red}`: 红蓝方二值化阈值（像素值，通常 80-120）
  - `detect.morphology_size.{blue,red}`: 形态学操作核大小（像素，通常 2-5，用于去噪）
  - `detect.dilate_size`: 膨胀操作次数（通常 1-3）
  - `R_contours_min_area`, `R_contours_max_area`: R 字母轮廓面积范围（像素²）
  - `fanblades_head_contours_min_area`, `fanblades_head_contours_max_area`: 扇叶头部轮廓面积范围（像素²）
  - `fanblades_body_contours_min_area`, `fanblades_body_contours_max_area`: 扇叶主体轮廓面积范围（像素²）
  - `standard_fanblade_path`: 标准扇叶模板图片路径，用于模板匹配
  - `fire_gap_time`: 两次开火之间的最小间隔（秒，防止连续过快开火）
  - `predict_time`: 预测时间窗口（秒，用于预测旋转目标未来位置）
  - `aim_time`: 预瞄时间（秒，部分配置中定义但未使用）
  - `wait_time`: 等待稳定时间（秒，部分配置中定义但未使用）
- 其他特定参数：`left_yaw_offset` / `right_yaw_offset` 用于双枪或左右区域补偿；`min_spin_speed` 区分静止与高速旋转策略。

### 2. `camera.yaml`
- 仅包含工业相机曝光、增益等基础成像控制参数；如果要切换不同品牌相机，可在此文件添加多配置并在运行时选择。
- 支持的相机类型：
  - `"hikrobot"`: 海康威视 USB 相机（使用 MVS SDK）
  - `"hikrobot_gige"`: 海康威视 GigE 相机（使用 GigE Vision 协议）
  - `"mindvision"`: 迈德威视相机（使用 MindVision SDK）

### 2.1 USB 相机参数详解（在部分 standard/sentry/uav 配置中出现）
- `image_width`, `image_height`（像素）：采集分辨率，驱动层通过 OpenCV V4L 设置到设备（`cv::CAP_PROP_FRAME_WIDTH/HEIGHT`）。使用位置：`io/usbcamera/usbcamera.cpp`。
- `usb_frame_rate`（FPS）：目标帧率（`cv::CAP_PROP_FPS`）。过高可能被设备/带宽限制回落。使用位置：`io/usbcamera/usbcamera.cpp`。
- `usb_exposure`（曝光，设备单位）：曝光时间设置（`cv::CAP_PROP_EXPOSURE`），范围通常 1-80000，同时代码设置 `cv::CAP_PROP_AUTO_EXPOSURE=1` 决定自动/手动模式，具体行为依设备驱动而定。使用位置：同上。
- `new_usb_exposure`（可选）：用于动态调整曝光值，部分配置中使用此参数在不同场景下切换曝光设置。
- `usb_gamma`：伽马（`cv::CAP_PROP_GAMMA`），用于整体亮度曲线调整。使用位置：同上。
- `usb_gain`：模拟/数字增益（`cv::CAP_PROP_GAIN`），提高亮度但会引入噪声。使用位置：同上。
- 设备名与多相机：代码读取 `cv::CAP_PROP_SHARPNESS` 判别左右相机并设置 `device_name`（left/right），仅用于日志与区分双目设备。使用位置：同上。
- `new_image_width`, `new_image_height`（像素，可选）：若存在，通常用于后续算法中的缩放/重映射尺寸（例如 YOLO 推理或可视化分辨率对齐），未被 `io/usbcamera` 直接消费。
- `fov_h`, `fov_v`, `new_fov_h`, `new_fov_v`（度，可选）：视场角参数，常用于将像素归一化到角度空间或多相机协同时的视角换算；参考使用位置：`tasks/omniperception/decider.*`（字段存在于头文件，实际计算通常在对应 cpp 中）。

注意事项：
- USB 相机依赖 V4L2 驱动能力，不同设备对 `EXPOSURE/GAMMA/GAIN` 支持与范围不同；设置失败时 OpenCV 可能静默回退，建议在日志中打印实际生效值。
- 分辨率/帧率越高越占用带宽，注意与 MJPG/H264 编码、主控 USB 控制器带宽的权衡。
- 若需要双目/多 USB 相机，请确保每个设备的 `/dev/video*` 名称与打开顺序稳定（可以结合 `udev` 规则固定 symlink）。

### 3. `calibration.yaml`
- `pattern_cols`, `pattern_rows`, `center_distance_mm`: 标定板网格规格（用于标定脚本计算内参）。
- 与部分运行配置重复的相机与 CAN 参数：可能用于标定阶段硬件初始化，与运行期配置区分（可合并或保持独立）。

### 4. `mvs.yaml`
- 多视觉系统（Multi-Vision System）配置文件，结构与其他配置文件相似
- 若扩展多视觉功能，可能需要添加的参数：
  - 多相机姿态参数（各相机相对于机体的旋转与平移）
  - 时间同步缓冲大小（用于多相机帧同步）
  - 融合策略模式（多视角数据融合方式）
  - 相机间标定参数（双目或多目视觉）

### 5. `example.yaml`
- 演示用简化版本：参数结构与标准配置相同，数值更易懂，适合作为新机型复制起点。

---
## 参数调试优先级建议
1. 先确保标定参数正确（重投影误差 < 0.5px）。
2. 调整曝光/增益保证目标不过曝或明显欠曝。
3. 根据场地光照与瑕疵调整 `threshold` 与光条/装甲板比率阈值，确认检测稳定性。
4. 确认跟踪器参数：降低误检跳变，防止丢失后频繁创建新目标。
5. 优化延迟相关：`high_speed_delay_time` 与 `low_speed_delay_time` 根据实时测得弹速与处理耗时微调。
6. MPC/规划权重微调：在离线日志上比较不同 `Q_*`、`R_*` 下角度误差与响应时间。
7. BUFF 任务：先固定亮度与形态参数再调预测时间窗，过大/过小都会导致提前或延后开火。

---
## 迁移到新平台的最小修改清单
- 相机：更新 `camera_name`, 曝光、分辨率、FOV；若无精确 FOV，可通过标定或测量更新。
- 标定：重新生成 `camera_matrix`, `distort_coeffs`, `R_camera2gimbal`, `t_camera2gimbal`。
- 云台/底盘：更新 `com_port`, CAN IDs；若协议不同需同步修改 `io/gimbal.*` / `io/cboard.*`。
- 模型：替换 `yolo*_model_path`，若类别数变动需修改后处理与 `classifier`。
- 任务裁剪：若仅自瞄，移除 BUFF 参数与模块；若仅 BUFF，精简 `auto_aim` 相关配置。
- 性能：根据算力将 `device` 设为 `GPU` / `CPU` / `"AUTO:NPU,CPU"`；使用更小输入尺寸或 int8 量化模型（如 `yolo11_buff_int8.xml`）。

---
## grep 快速定位示例命令
```bash
# 查找参数在源码中的引用（Linux shell 示例）
grep -R "yaw_offset" -n ./tasks/auto_aim ./src
grep -R "fire_thresh" -n ./tasks/auto_aim/planner ./src
grep -R "R_camera2gimbal" -n ./tasks/auto_aim ./tools
```

---
## 维护建议
- 新增参数：请在对应 YAML 下方加入注释并同步更新此文件一行说明；命名保持语义清晰（单位、作用）。
- 删除参数：先搜索代码引用确保无遗留，再更新此文件移除条目。
- 单位约定：角度参数统一注释“degree”，时间统一“s”，距离“m”，亮度调整为绝对数值或归一到 [0,255]。

---
---

## 参数单位与取值范围参考

### 角度参数
- 单位：度（degree），代码中会转换为弧度（rad）
- 典型范围：
  - `yaw_offset`, `pitch_offset`: -10° 到 10°
  - `comming_angle`: 50° 到 70°
  - `leaving_angle`: 15° 到 30°
  - `first_tolerance`, `second_tolerance`: 0.5° 到 5°

### 时间参数
- 单位：秒（s）
- 典型范围：
  - `high_speed_delay_time`, `low_speed_delay_time`: 0.005s 到 0.1s
  - `fire_gap_time`: 0.5s 到 0.7s
  - `predict_time`: 0.09s 到 0.3s
  - `exposure_ms`: 0.8ms 到 8ms（工业相机）

### 距离参数
- 单位：米（m）
- 典型范围：
  - `judge_distance`: 0.5m 到 3m

### 角速度参数
- 单位：弧度每秒（rad/s）
- 典型范围：
  - `decision_speed`: 7 rad/s 到 12 rad/s
  - `min_spin_speed`: 2 rad/s 到 5 rad/s

### 图像处理参数
- `threshold`: 像素值，通常 100-200
- `min_confidence`: 置信度，0.0-1.0，通常 0.7-0.9
- `detect.brightness.{blue,red}`: 亮度偏移，通常 -150 到 0
- `detect.brightness_threshold.{blue,red}`: 二值化阈值，通常 80-150

### MPC 参数
- `fire_thresh`: 弧度（rad），通常 0.003-0.004
- `max_yaw_acc`, `max_pitch_acc`: 弧度/秒²（rad/s²），通常 50-100
- `Q_yaw`, `Q_pitch`: 状态权重矩阵，通常 `[9e6, 0]` 或 `[1e6, 0]`
- `R_yaw`, `R_pitch`: 控制权重，通常 `[1]`

---

## 常见问题与调试技巧

### 1. 检测不稳定
- 检查 `threshold` 是否适合当前光照条件
- 调整 `min_lightbar_ratio`, `max_lightbar_ratio` 过滤误检
- 检查 `min_detect_count` 是否过小（建议 5-10）
- 确认 `exposure_ms` 和 `gain` 设置合理，避免过曝或欠曝

### 2. 跟踪丢失频繁
- 增大 `max_temp_lost_count`（但不要过大，避免跟踪错误目标）
- 对于前哨站等特殊目标，调整 `outpost_max_temp_lost_count`
- 检查 ROI 设置是否合理（`use_roi` 和 `roi` 参数）

### 3. 射击精度问题
- 首先确认标定参数正确（重投影误差 < 0.5px）
- 调整 `yaw_offset`, `pitch_offset` 补偿安装误差
- 根据实际弹速调整 `high_speed_delay_time`, `low_speed_delay_time`
- 检查 `first_tolerance`, `second_tolerance` 是否过宽松

### 4. BUFF 检测失败
- 调整 `detect.brightness.{blue,red}` 和 `detect.brightness_threshold.{blue,red}` 适应光照
- 检查轮廓面积参数是否适合当前距离和视角
- 确认 `model` 路径正确且模型文件存在
- 调整 `detect.morphology_size` 和 `detect.dilate_size` 优化图像预处理

### 5. 性能优化
- 使用 ROI 裁剪减少处理区域（`use_roi: true`）
- 选择更小的模型或 int8 量化模型
- 根据硬件选择合适推理设备（NPU > GPU > CPU）
- 降低 USB 相机分辨率或帧率（`image_width`, `image_height`, `usb_frame_rate`）

### 6. 多相机配置
- 使用 `vid_pid` 区分不同 USB 相机设备
- 确保每个相机有独立的标定参数（`camera_matrix`, `distort_coeffs`, `R_camera2gimbal`, `t_camera2gimbal`）
- 设置正确的 FOV 参数用于多视角融合

---

最后更新日期：2025-11-14
