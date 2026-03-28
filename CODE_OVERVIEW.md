# 项目代码概览 (Code Overview)

本文件用于快速了解 `rm_vision_2025` 仓库中各目录及代码文件的职责，便于新人上手与后续维护。描述力求：一句话概括 + 关键词。需要更详细的接口说明时，请直接阅读对应头文件或实现文件。

> 约定：`自瞄` = RoboMaster 装甲板识别与击打；`能量机关/BUFF` = Power Rune 识别；`云台` = Gimbal；`MPC` = Model Predictive Control；`YOLO` = 目标检测神经网络。

---
## 项目作用概述
`rm_vision_2025` 旨在为 RoboMaster 等对抗机器人平台提供一套高性能、模块化、可扩展的视觉与决策系统，覆盖自瞄、能量机关（BUFF）击打、哨兵防御、无人机视觉协同等典型任务。它集成多种相机与姿态传感器输入，基于深度学习与几何算法实时识别目标，并结合物理与优化控制（MPC、弹道解算）输出精确的云台控制与射击决策。

关键词：实时性、鲁棒性、多源融合、模块化、可移植性、可调试性。

## 典型应用场景
- 赛事对抗：在 RoboMaster 比赛中执行不同机器人角色（标准、哨兵、无人机）视觉打击与防御策略。
- 能量机关风车击打：识别旋转扇叶与“R”字中心，预测最佳开火时刻。
- 多相机协同：前后/左右/顶部多个视角整合，提高遮挡场景下目标保持率。
- 策略研究与算法验证：MPC、ADMM 优化、EKF 跟踪、RANSAC 拟合等算法在真实物理场景下的性能测试。
2. 工具与算法基础（`tools/`）：提供通用的数学、滤波、并发、日志、数据记录与拟合组件，支撑上层任务无需重复造轮子。
## 学习路径与上手顺序（建议）
面向新同学的循序渐进路线，建议按序完成，每步确保可运行再继续：

1. 快速鸟瞰（0.5 天）
  - 通读本文件与 `readme.md`，了解模块分层与入口程序（`src/*.cpp`）。
  - 浏览 `configs/*.yaml`，认识关键参数（相机内外参、模式、阈值）。

2. 环境与构建（0.5~1 天）
  - 安装依赖（OpenCV、Eigen、YAML-CPP、串口/SocketCAN、可选 OpenVINO/ONNX Runtime）。
  - 本地编译并运行一个最小可执行：`tests/minimum_vision_system.cpp` 或 `src/standard.cpp`（离线素材或简单相机源）。

3. 标定与验证（1 天）
  - 跑通 `calibration/` 下相机标定和手眼标定，产出 `calibration.yaml`。
  - 用 `tests/handeye_test.cpp` 和投影重投影误差验证标定正确性。

4. IO 层理解与替换（1 天）
  - 研读 `io/`：`camera.*`、`usbcamera.*`、`gimbal.*`、`cboard.*`、`serial/`。
  - 使用 `tests/camera_test.cpp`、`tests/gimbal_test.cpp`、`tests/serial_test.cpp` 分别验收硬件通信。

5. 自瞄管线（2 天）
  - 顺序阅读 `tasks/auto_aim/`：`armor` → `detector` → `yolo` → `tracker/target` → `solver` → `aimer/shooter`。
  - 用 `tests/auto_aim_test.cpp`、`tests/detector_video_test.cpp` 离线验证，再接入实时相机。

6. BUFF 任务（1~2 天）
  - 阅读 `tasks/auto_buff/`：`buff_detector` → `buff_solver` → `buff_aimer`。
  - 用 `tests/auto_buff_test.cpp` 或自制素材验证时序预测与命中窗口。

7. 多源融合与策略（1 天）
  - 阅读 `tasks/omniperception/decider.*`，理解优先级映射与队列排序。
  - 在多相机/多任务下调参（FOV、裁剪、优先级表）。

8. 规划与控制（1~2 天）
  - 研读 `planner/`（含 `tinympc`）、`tools/trajectory.*`、`tools/pid.*`。
  - 用 `tests/planner_test*.cpp` 验证在线/离线 MPC；与弹道解算、PID 联调。

9. 综合联调与性能（持续）
  - 启用多线程版本（`mt_*`、`multithread/`）、切换轻量模型（`assets/` int8）。
  - 关注日志与帧率、时延分布，逐步优化。

提示：每阶段保留一套“可回滚”的配置快照（YAML + 模型 + 标定文件）。

---
## 迁移与自定义指南（将本项目用到你的机器人）
目标：在新硬件/新策略上稳定运行。建议按以下清单实施：

1) 明确硬件与接口
- 相机：分辨率、帧率、驱动（USB/HIK 等）。若为新品牌，参考 `io/usbcamera` 增加派生实现。
- 云台/底盘：串口协议或 CAN 报文格式，若不兼容，修改 `io/gimbal.*` 或 `io/cboard.*` 的收发结构体并同步 CRC。
- IMU：姿态四元数来源与时间戳对齐，必要时在 `io/` 增加适配层。

2) 标定与外参
- 通过 `calibration/` 生成新的 `calibration.yaml`（相机内外参、畸变）。
- 在 `tasks/auto_aim/solver.*` 中确保坐标变换（`R_camera2gimbal`, `t_camera2gimbal`, `R_gimbal2world`）与实际一致。

3) 配置与模型
- 复制并重命名一份 `configs/example.yaml` 为你的机型配置；调整相机话题/分辨率、阈值、模式、优先级表。
- 替换/量化模型：将你的检测模型放入 `assets/`，在相应 `yolo*` 适配器或 YAML 中指向新路径；注意输入尺寸与类别映射。

4) 管线裁剪与扩展
- 仅自瞄：保留 `tasks/auto_aim`，在 `src/standard.cpp` 基础上精简。
- 新目标类别：扩展 `armor::ArmorName` 和 `classifier`，补充数据集与训练/量化流程。
- 新策略/模式：在 `src/` 新建入口（可参考 `sentry.cpp`、`uav.cpp`），或在 `omniperception::Decider` 中新增优先级映射。

5) 控制回路与时延补偿
- 先用 `tools/pid` 打通闭环，再逐步接入 `planner/tinympc` 做前瞻控制。
- 用 `tools/trajectory` 与实测弹道拟合参数，修正落点。
- 在 `io/gimbal` 中对齐时间戳，使用四元数队列插值 `q(t)`，避免姿态-图像错位。

6) 性能与稳定性
- 开多线程与生产者-消费者队列（`multithread/`、`tools/thread_safe_queue`）。
- 降分辨率、裁剪 ROI、切换 int8 模型，优先保障 60FPS+ 与端到端时延。
- 记录日志与关键帧图，复盘异常时刻。

7) 验收与回归
- 逐层运行对应 `tests/*.cpp`，从 IO → 检测 → 解算 → 控制 全链路打通。
- 固定一套场景素材做回归基线，避免参数漂移。

小贴士：修改公共接口时，优先在头文件加入文档注释，避免团队沟通成本；变更 YAML 字段时在此文件追加说明。

---
## 常见坑与排查清单
- 时间同步：图像时间戳与 IMU/云台姿态错位 → 使用插值队列并统一时基。
- 坐标系方向：yaw/pitch 正负号或轴定义不一致 → 统一右手系并在 `solver` 验证重投影误差。
- 标定失配：分辨率变更但仍用旧内参 → 重新标定或做内参缩放适配。
- 通信权限：串口/CAN 无权限或断连 → udev 规则、波特率/位宽校准、重连机制。
- 模型不匹配：输入尺寸/通道/类别映射不一致 → 在 `yolo` 适配器中显式检查并报错。
- 性能瓶颈：拷贝、显示阻塞、单线程后处理 → 使用零拷贝、批处理、线程池。
- 量化精度损失：int8 精度掉点 → 重新校准量化集或回退 FP16/FP32。

4. 感知与融合（`omniperception`）：整合多源检测结果，执行优先级重新分配、模式切换、目标筛选，输出统一决策命令。
5. 规划与控制（MPC + 弹道 + PID）：利用 `planner/` 与物理模型（`trajectory`）结合预测未来目标状态，补偿延迟与弹道下坠，提升命中率。
6. 顶层入口（`src/`）：根据运行模式加载对应配置、选择任务管线、启动线程与主循环。

关键技术要点：
- 深度学习 + 几何混合：YOLO 获取候选区域，传统几何与光条/角点校正提升精度。
- 多目标管理：跟踪器与投票器结合，降低误检与抖动影响。
- 时序与姿态对齐：利用四元数插值与时间戳同步 IMU / 云台姿态与图像帧。
- 鲁棒控制：弹道解算 + PID/MPC 双层，兼顾快速响应与预测补偿。
- 并行与性能优化：多线程检测、命令生成、轻量化模型（int8/ONNX/OpenVINO）。
- 可配置性：所有阈值、模式优先级、相机/模型选择通过 YAML 动态调整。

## 与其他常见方案的差异
- 深度融合传统视觉几何逻辑（光条、装甲板角点重投影优化），避免纯深度模型在比赛极端光照下失效。
- 内置轻量 MPC（`tinympc`）用于未来姿态与角度规划，而非仅依赖简单比例或前馈控制。
- 提供多角色入口程序与统一命令结构，支持快速在不同机器人平台间迁移。

## 扩展方向建议
- 引入多目标协同策略（装甲板 + BUFF 任务动态切换）。
- 增加自适应曝光 / HDR 预处理提高逆光鲁棒性。
- 利用 GPU/VPU 推理加速与算力动态调度。
- 引入学习型跟踪器（如基于 Transformer 的时序增强）替代传统滤波器。
- 增加自动生成 `CODE_OVERVIEW.md` 的脚本（遍历头文件注释）。

---

---
## 根目录
- `CMakeLists.txt`：顶层构建配置，聚合各子模块、第三方依赖与编译选项。
- `autostart.sh`：启动脚本，可能用于部署环境自动拉起视觉进程。
- `buff_layout.xml` / `mpc_layout.xml`：推测为调试或 UI/配置布局文件（如可视化工具的界面布局）。
- `readme.md`：项目总体介绍（主文档）。
- `LICENSE`：开源许可。
- `.clang-format`：代码格式规则。

---
## `assets/`
存放推理所需模型/权重与演示资源：
- `best2-sim.onnx` / `tiny_resnet.onnx`：分类或特征提取网络。
- `yolo11_buff_int8.xml` / `yolo11.xml` / `yolov5.xml` / `yolov8.xml`：不同版本 YOLO 模型（OpenVINO IR / ONNX 等）。
- `demo/`：演示素材（推测为测试视频或图片）。

---
## `configs/`
各模式/平台的 YAML 参数：
- `camera.yaml`：相机参数（分辨率、曝光、硬件配置）。
- `calibration.yaml`：标定参数（内参与外参、畸变系数）。
- `sentry.yaml` / `uav.yaml` / `standard3.yaml` / `standard4.yaml` 等：不同机器人或比赛模式的运行参数（敌方颜色、策略阈值、优先级设置等）。
- `example.yaml` / `demo.yaml`：示例或演示用配置。
- `mvs.yaml`：多视觉/多传感源相关参数。
- `ascento.yaml`：可能对应特定平台或测试场景。

---
## `calibration/`
标定/辅助工具：
- `calibrate_camera.cpp`：单相机标定（棋盘格/AprilTag）。
- `calibrate_handeye.cpp`：手眼标定（相机与机械结构坐标系关系）。
- `calibrate_robotworld_handeye.cpp`：Robot-World/Hand-Eye 联合标定。
- `capture.cpp`：采集标定或训练数据帧。
- `split_video.cpp`：视频拆分（分帧或裁剪）。

---
## `io/` 输入输出与硬件接口层
- `CMakeLists.txt`：此目录子模块构建配置。
- `camera.hpp / camera.cpp`：抽象与封装多品牌/类型相机读取流程（时间戳、图像获取）。
- `cboard.hpp / cboard.cpp`：控制板通信（CAN），维护 IMU 四元数、弹速、模式切换等；含异步队列防止阻塞。
- `gimbal/gimbal.hpp / gimbal.cpp`：云台串口协议 (Vision ↔ Gimbal)，姿态、速度、弹速读取与控制指令发送。
- `dm_imu/dm_imu.hpp / dm_imu.cpp`：地面或独立 IMU 设备数据接入。
- `usbcamera/usbcamera.hpp / usbcamera.cpp`：USB 相机具体实现。
- `hikrobot/hikrobot.cpp / hikrobot.hpp`：海康相机驱动封装。
- `socketcan.hpp`：Linux SocketCAN 接口适配。
- `command.hpp`：项目内部标准化“指令/控制包”定义。
- `serial/`：跨平台串口库（`serial.h / serial.cc`）及端口枚举实现（`list_ports_*`）。
- `ros2/`：（未展开）可能是 ROS2 话题封装或桥接层。

---
## `tools/` 通用工具与算法库
- `logger.hpp / logger.cpp`：日志系统（级别、格式、线程安全）。
- `thread_safe_queue.hpp`：有界/无界线程安全队列（用于传感器数据/消息传递）。
- `thread_pool.hpp`：线程池执行任务调度。
- `img_tools.hpp / img_tools.cpp`：图像预处理、ROI、阈值/形态操作辅助函数。
- `math_tools.hpp / math_tools.cpp`：几何/矩阵/角度/插值等数学辅助。
- `pid.hpp / pid.cpp`：基本 PID 控制器，支持角度 wrap 处理。
- `trajectory.hpp / trajectory.cpp`：弹道解算（给定初速、水平距离、高度 → 抬头角与飞行时间）。
- `extended_kalman_filter.hpp / extended_kalman_filter.cpp`：通用 EKF（支持自定义状态/测量组合、NIS/NEES 统计）。
- `crc.hpp / crc.cpp`：通信校验 (CRC16/CRC32)。
- `plotter.hpp / plotter.cpp`：数据绘制或调试可视化。
- `recorder.hpp / recorder.cpp`：数据记录（图像/状态流）。
- `ransac_sine_fitter.hpp / ransac_sine_fitter.cpp`：RANSAC 正弦曲线拟合（周期性目标轨迹 / 风车叶片）。
- `exiter.hpp / exiter.cpp`：优雅退出钩子（信号处理）。
- `yaml.hpp`：YAML 加载封装/类型转换辅助。

---
## `tasks/auto_aim/` 自瞄主逻辑
- `CMakeLists.txt`：模块构建。
- `aimer.hpp / aimer.cpp`：整体自瞄流程协调（检测→跟踪→解算→发送指令）。
- `armor.hpp / armor.cpp`：装甲板数据结构（角点、类型、编号、姿态、跟踪状态）。
- `classifier.hpp / classifier.cpp`：装甲板数字/类别识别（可能使用轻量 CNN / 模板匹配）。
- `detector.hpp / detector.cpp`：传统/几何自瞄装甲板检测（光条提取、阈值、形态校验、图像特征）。
- `yolo.hpp / yolo.cpp`：YOLO 检测封装（推理 + 后处理 → 装甲板候选）。
- `yolos/`：不同 YOLO 版本适配 (`yolo11.cpp/yolov5.cpp/yolov8.cpp`)。
- `target.hpp / target.cpp`：目标跟踪/过滤（状态预测、优先级）。
- `tracker.hpp / tracker.cpp`：单/多目标跟踪器（可能用卡尔曼或 EKF）。
- `solver.hpp / solver.cpp`：位姿与角度解算（PnP、重投影误差优化、云台坐标系转换）。
- `voter.hpp / voter.cpp`：多检测/多策略结果投票融合。
- `shooter.hpp / shooter.cpp`：击发决策（时机、弹速补偿、延迟估计）。
- `multithread/mt_detector.hpp / mt_detector.cpp`：多线程检测加速。
- `multithread/commandgener.hpp / commandgener.cpp`：并行生成控制指令。
- `planner/`：MPC/优化控制：
  - `planner.hpp / planner.cpp`：自瞄轨迹/控制规划入口。
  - `tinympc/`：轻量 MPC 算法：`admm.*`、`tiny_api.*`、`rho_benchmark.*`、`codegen.*`、`types.hpp`、`error.hpp`、`tiny_api_constants.hpp` 等实现迭代优化与参数基准。

---
## `tasks/auto_buff/` 能量机关识别与预测
- `buff_detector.hpp / buff_detector.cpp`：符识别主流程（风车/能量机关装甲版/扇叶提取、状态管理、丢失处理）。
- `buff_aimer.hpp / buff_aimer.cpp`：能量机关瞄准策略（选取最佳击打窗口）。
- `buff_solver.hpp / buff_solver.cpp`：位姿/弹道/时间预测解算。
- `buff_target.hpp / buff_target.cpp`：目标状态封装与更新。
- `buff_type.hpp / buff_type.cpp`：风车结构/扇叶/“R”识别类型定义。
- `buff_predict.hpp`：预测相关结构或函数声明（仅头文件）。
- `yolo11_buff.hpp / yolo11_buff.cpp`：YOLO11 专用于 BUFF 检测的推理封装。

---
## `tasks/omniperception/` 多源融合与决策
- `decider.hpp / decider.cpp`：多相机/多任务感知结果融合，生成最终 `io::Command`；包含优先级、模式切换与装甲板筛选。
- `perceptron.hpp / perceptron.cpp`：感知管线抽象（图像输入→检测→融合）。
- `detection.hpp`：检测结果数据结构与接口定义。

---
## `src/` 顶层程序入口与模式实现
这些 `*.cpp` 多为不同运行模式主程序 (main 或 orchestrator)：
- `standard.cpp / standard_mpc.cpp`：标准机器人模式（含 MPC 控制版本）。
- `auto_aim_debug_mpc.cpp` / `mt_auto_aim_debug.cpp`：自瞄调试（多线程/MPC）。
- `auto_buff_debug.cpp / auto_buff_debug_mpc.cpp`：能量机关调试脚本。
- `sentry.cpp / sentry_debug.cpp / sentry_multithread.cpp / sentry_bp.cpp`：哨兵模式及变体（多线程、基准测试、不同策略）。
- `uav.cpp / uav_debug.cpp`：无人机视觉模式（空中平台）。

---
## `tests/` 功能/集成测试与示例
- `auto_aim_test.cpp`：自瞄流水线验证。
- `auto_buff_test.cpp`：能量机关检测预测验证。
- `camera_test.cpp` / `camera_thread_test.cpp`：相机读取与多线程性能测试。
- `camera_detect_test.cpp`：相机 + 检测集成。
- `cboard_test.cpp`：控制板通信测试。
- `gimbal_test.cpp` / `gimbal_response_test.cpp`：云台控制与响应延迟评估。
- `dm_test.cpp`：IMU/数据融合测试。
- `planner_test.cpp / planner_test_offline.cpp`：MPC 规划在线/离线验证。
- `multi_usbcamera_test.cpp`：多 USB 相机并行采集测试。
- `serial_test.cpp`：串口读写测试。
- `usbcamera_test.cpp / usbcamera_detect_test.cpp`：USB 相机 + 检测组合。
- `topic_loop_test.cpp / publish_test.cpp / subscribe_test.cpp`：若集成 ROS2，则是话题发布/订阅循环测试。
- `minimum_vision_system.cpp`：最小化可运行示例（基础自瞄或感知链）。
- `fire_test.cpp`：击发逻辑/延迟测试。
- `handeye_test.cpp`：手眼标定结果验证。
- `detector_video_test.cpp`：离线视频检测回放。

---
## 其他目录
- `patterns/`：可能存放模板图案或二值化匹配资源。
- `imgs/`：图像资源（标定、调试截图、示例输入）。
- `logs/`：运行生成日志目录（空占位）。
- `Build/` / `build/`：构建输出目录（区分大小写环境下可能保留两个）。

---
## 典型调用链示例
1. 入口程序（例如 `standard.cpp`）加载 `configs/standard*.yaml`。
2. 初始化硬件：`io::Camera` / `io::Gimbal` / `io::CBoard`。
3. 感知：`tasks/auto_aim::YOLO` 或 `Detector` → 构造 `Armor` 列表。
4. 追踪与解算：`Tracker` / `Solver` / `Trajectory` / `PID`。
5. 决策：`aimer` / `shooter` / 可能通过 `omniperception::Decider` 融合多源。
6. 输出控制：封装为 `io::Command` → 发送至云台或控制板 (`CBoard::send`, `Gimbal::send`)。

---
## 更新维护建议
- 新增文件：请在创建时追加到本 `CODE_OVERVIEW.md` 相应章节。
- 大改接口：同步更新描述中的关键词（如增加 EKF 状态维度、修改 YOLO 版本）。
- 可考虑后续自动生成：脚本遍历 `tasks/*` 读取首行注释并更新此文档。

---
## 附：尚未深入的文件
部分 `.cpp` 未展开阅读，描述基于命名与上下文推断；如需精准定位，请直接查阅源码或补充注释。

若发现错误或需要更细粒度说明（例如各类枚举含义、坐标系转换公式、MPC 状态向量定义），欢迎在对应头文件顶部添加简短文档块，并在此处引用。

---
最后更新日期：2025-11-13
