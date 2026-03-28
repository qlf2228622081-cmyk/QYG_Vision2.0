# USB2CAN — slcand 与 SocketCAN 完整操作指南

本指南涵盖如何把常见 USB2CAN 设备通过 slcand 绑定为 SocketCAN 接口（如 `can0`）、常用调试命令、systemd/udev 模板、权限与故障排查。目标是能让你在调试和生产环境都能可靠启动并排查问题。

## 目录

- [前提](#前提)
- [1. 识别设备与基础排查](#1-识别设备与基础排查)
- [2. 使用 slcand 将串口绑定为 SocketCAN](#2-使用-slcand-将串口绑定为-socketcan推荐)
- [3. 常用 SocketCAN 调试命令](#3-常用-socketcan-调试命令)
- [4. systemd 服务示例](#4-systemd-服务示例开机自动启动)
- [5. udev 规则与权限](#5-udev-规则与权限推荐)
- [6. 权限与常见问题](#6-权限与常见问题)
- [7. 故障排查与调试技巧](#7-故障排查与调试技巧)
- [8. 回收与卸载 slcand](#8-回收与卸载-slcand)
- [9. 建议的工作流程](#9-建议的工作流程调试阶段)
- [10. 示例：把模板放到仓库并部署](#10-示例把模板放到仓库并部署)
- [11. 与项目代码集成](#11-与项目代码集成)
- [12. SocketCAN vs USB2CAN SDK 对比](#12-socketcan-vs-usb2can-sdk-对比)
- [13. CAN 位速率（Bitrate）选择](#13-can-位速率bitrate选择)
- [14. 回环模式测试](#14-回环模式测试)
- [15. 多设备场景](#15-多设备场景)
- [16. 监控与统计](#16-监控与统计)
- [17. 性能调优建议](#17-性能调优建议)
- [18. 常见错误与解决方案](#18-常见错误与解决方案)
- [附：快速命令小结](#附快速命令小结)

## 前提

- 运行 Linux（示例基于 Debian/Ubuntu，命令在其他发行版上通常类似）。
- 安装 must-have 工具：

```bash
sudo apt update
sudo apt install can-utils usbutils
```

`can-utils` 包含 `slcand`、`candump`、`cansend` 等工具，`usbutils` 提供 `lsusb`。

## 1. 识别设备与基础排查

- 查看内核日志识别新插入设备：

```bash
dmesg | tail -n 50
ls -l /dev/ttyUSB* /dev/ttyACM* /dev/serial/by-id/*
```

**示例输出**：
```
[12345.678901] usb 1-1.2: new full-speed USB device number 5 using ehci-pci
[12345.789012] usb 1-1.2: New USB device found, idVendor=1a86, idProduct=7523
[12345.789013] usb 1-1.2: New USB device strings: Mfr=1, Product=2, SerialNumber=0
[12345.789014] usb 1-1.2: Product: USB2.0-Serial
[12345.890123] ch341 1-1.2:1.0: ch341-uart converter detected
[12345.901234] usb 1-1.2: ch341-uart converter now attached to ttyUSB0
```

- 使用 `lsusb` 找到设备的 `idVendor:idProduct`（用于 udev 规则或确认设备型号）：

```bash
lsusb
# 例子输出: Bus 001 Device 005: ID 1a86:7523 QinHeng Electronics HL-340
```

- 使用 `udevadm` 获取设备详细信息：

```bash
# 获取设备详细信息（替换为实际设备路径）
udevadm info /dev/ttyUSB0

# 或通过 by-id 路径
udevadm info /dev/serial/by-id/usb-*-if00
```

**示例输出**：
```
P: /devices/pci0000:00/0000:00:1d.0/usb1/1-1/1-1.2/1-1.2:1.0/ttyUSB0/tty/ttyUSB0
N: ttyUSB0
S: serial/by-id/usb-1a86_USB2.0-Serial-if00-port0
E: DEVPATH=/devices/pci0000:00/0000:00:1d.0/usb1/1-1/1-1.2/1-1.2:1.0/ttyUSB0/tty/ttyUSB0
E: ID_VENDOR_ID=1a86
E: ID_MODEL_ID=7523
```

- 如果设备被 SDK 或固件控制，需要参考 `USB2CAN_SDK` 中示例（例如 `USB2CAN_SDK/example/demo_usb2can.py`）来确认设备是否需要切换到 SLCAN 模式。

## 2. 使用 `slcand` 将串口绑定为 SocketCAN（推荐）

常见流程：先用 `slcand` 将串口映射为 `can0` 然后用 `ip` 启动并设置 bitrate。

示例：

```bash
# 把 /dev/ttyUSB0 通过 slcand 绑定为 can0（指定串口波特率 115200）
sudo slcand -S 115200 -o -c /dev/ttyUSB0 can0

# 然后设置 CAN 位速并启动接口
sudo ip link set can0 up type can bitrate 1000000

# 检查接口状态
ip -details link show can0
ip addr show can0
```

**正常输出示例**：
```
3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UNKNOWN mode DEFAULT group default qlen 10
    link/can  promiscuity 0
    can <TRIPLE-SAMPLING> state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0
          bitrate 500000 sample-point 0.875
          tq 125 prop-seg 6 phase-seg1 7 phase-seg2 2 sjw 1
          slcand: ttyUSB0 at 115200
    RX: bytes  packets  errors  dropped overrun mcast
            0        0       0        0       0     0
    TX: bytes  packets  errors  dropped carrier collsns
            0        0       0        0       0       0
```

**关键状态说明**：
- `UP`：接口已启动
- `ERROR-ACTIVE`：CAN 控制器处于正常状态（无错误）
- `berr-counter tx 0 rx 0`：总线错误计数为 0（正常）
- `bitrate 500000`：当前 bitrate 为 500 kbps

说明：
- `-S 115200`：设置串口波特率为 115200（有些设备需要 921600 等，请参考设备文档）。
- `-o`/`-c`：常用选项，具体见 `man slcand`。不同固件可能需要不同选项（例如是否发送开机命令）。

如果你更愿意一次性完成绑定并启动，也可把 `ip link set ...` 放到 `slcand` 的后续命令或 systemd 的 `ExecStartPost`。

注意：在某些发行版 `slcand` 的路径在 `/sbin` 或 `/usr/sbin`，systemd 配置中请使用绝对路径。

## 3. 常用 SocketCAN 调试命令

- 监听（实时）：

```bash
candump can0
```

**输出示例**：
```
can0  123   [4]  11 22 33 44
can0  456   [8]  AA BB CC DD EE FF 00 11
can0  789   [2]  22 33
```

**格式说明**：`接口名  CAN_ID  [数据长度]  数据字节（十六进制）`

- 发送（十六进制）：

```bash
# 标准帧，ID=0x123，数据=11 22 33 44
cansend can0 123#11223344

# 扩展帧，ID=0x12345678
cansend can0 12345678#11223344

# 远程帧请求
cansend can0 123#R
```

- 过滤特定 CAN ID：

```bash
# 只监听 ID 0x123
candump can0,123:7FF

# 监听 ID 范围 0x100-0x1FF
candump can0,100:1FF
```

- 抓包并保存以便离线分析：

```bash
candump -L can0 > can0.log
```

- 停用接口：

```bash
sudo ip link set down can0
# 如需要，删除接口（如果是 slcand 创建的）
sudo ip link delete can0
```

## 4. systemd 服务示例（开机自动启动）

建议使用设备的持久 symlink（`/dev/serial/by-id/...`）或 udev 创建的固定名，而不要直接写 `/dev/ttyUSB0`（设备编号可能变）。

**如何找到设备的 by-id 路径**：

```bash
# 方法1：列出所有串口设备
ls -l /dev/serial/by-id/

# 方法2：通过 udevadm 查找
udevadm info /dev/ttyUSB0 | grep SERIAL_SHORT

# 方法3：查看符号链接
readlink -f /dev/serial/by-id/*
```

**示例输出**：
```
lrwxrwxrwx 1 root root 13 Jan  1 12:00 usb-1a86_USB2.0-Serial-if00-port0 -> ../../ttyUSB0
```

示例 service `/etc/systemd/system/slcand-usb2can.service`：

```ini
[Unit]
Description=slcand for USB2CAN (by-id)
After=network.target
# 可选：等待设备就绪
BindsTo=dev-serial-by\x2did-usb\x2d1a86_USB2.0\x2dSerial\x2dif00\x2dport0.device
After=dev-serial-by\x2did-usb\x2d1a86_USB2.0\x2dSerial\x2dif00\x2dport0.device

[Service]
Type=simple
# 使用 by-id 路径替代 /dev/ttyUSB0
# 注意：路径中的特殊字符需要转义（如 - 转义为 \x2d）
ExecStart=/usr/sbin/slcand -S 115200 -o -c /dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0 can0
ExecStartPost=/sbin/ip link set can0 up type can bitrate 500000
# 可选：设置重启延迟
RestartSec=5
Restart=on-failure
# 可选：日志输出
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**注意**：如果使用 `BindsTo`，需要将设备路径中的特殊字符转义：
- `-` → `\x2d`
- `/` → `\x2f`
- 空格 → `\x20`

**简化版本**（不等待设备，依赖 Restart 机制）：

```ini
[Unit]
Description=slcand for USB2CAN
After=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/slcand -S 115200 -o -c /dev/serial/by-id/usb-1a86_USB2.0-Serial-if00-port0 can0
ExecStartPost=/sbin/ip link set can0 up type can bitrate 500000
RestartSec=5
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now slcand-usb2can.service
sudo systemctl status slcand-usb2can.service
```

如果设备节点可能会变化，请用 udev 规则或 by-id 路径确保稳定。

## 5. udev 规则与权限（推荐）

要避免每次 sudo，可以把设备权限或 symlink 固定下来：

1) 找到设备 idVendor/idProduct：

```bash
lsusb
# 或者从 dmesg 查找对应 tty
dmesg | grep tty
```

2) 示例 udev 规则 `/etc/udev/rules.d/99-usb2can.rules`：

```udev
# 将下面的 idVendor/idProduct 替换为实际值
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="usb2can_%k", GROUP="dialout", MODE="0660"
```

重载规则并触发：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

也可以直接使用 `/dev/serial/by-id/<id>` 作为 systemd 的设备路径。

## 6. 权限与常见问题

- 如果出现权限问题，把当前用户加入 `dialout` 组（或对应组）：

```bash
sudo usermod -a -G dialout $USER
# 之后重新登录会话
```

- slcand 立即退出：检查串口权限与串口波特率（`-S`），并使用 `strace` 或 `systemd` 日志查看错误。
- 无法收到数据：确认 CAN 总线两端终端电阻、波特率一致；使用两个节点直连回环测试。
- 接口名反复变化：使用 udev 或 by-id 路径。

## 7. 故障排查与调试技巧

- 打开详细日志：

```bash
sudo slcand -v -S 115200 -o -c /dev/ttyUSB0 can0
```

- 打印原始串口字节（帮助分析解析问题）：

```bash
sudo hexdump -C -n 64 /dev/ttyUSB0
```

- 如果使用 `USB2CAN_SDK`（项目内含 `USB2CAN_SDK` 目录），有两种方式：
	- 设备本身支持 SLCAN：更适合使用 `slcand` + SocketCAN 工具链（便于与现有 ROS / can-utils 集成）。
	- 设备通过厂商 API（SDK）访问 CAN：直接使用 SDK 进行收发（绕过 slcand）。参考 `USB2CAN_SDK/example/demo_usb2can.py`。

## 8. 回收与卸载 slcand

```bash
# 找到并终止 slcand
ps aux | grep slcand
sudo pkill slcand

# 如果需要，删除 can0
sudo ip link delete can0
```

## 9. 建议的工作流程（调试阶段）

1. 插入设备，`dmesg` 确认设备节点。
2. 使用 `lsusb` / `udevadm info` 确认 idVendor/idProduct 与 by-id 名称。
3. 先在命令行手动运行 `slcand` + `ip link set up`，用 `candump`/`cansend` 验证连通性。
4. 根据需要写 udev 规则 和 systemd 服务，放到仓库 `configs/` 或 `tools/` 目录以便部署。

## 10. 示例：把模板放到仓库并部署

如果你希望我把 systemd 单元文件和 udev 规则加入到仓库（例如 `configs/slcand-usb2can.service` 与 `configs/99-usb2can.rules`），我可以帮你生成并提交。然后你可以按下面步骤部署：

```bash
# 把规则复制到 /etc/udev/rules.d/
sudo cp configs/99-usb2can.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger

# 把 systemd 单元复制并启用
sudo cp configs/slcand-usb2can.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now slcand-usb2can.service
```

## 11. 与项目代码集成

本项目（`rm_vision_2025`）支持两种 CAN 接口方式：

### 11.1 SocketCAN 方式（推荐）

在配置文件中指定 SocketCAN 接口名：

```yaml
# configs/example.yaml
can_interface: "can0"  # 通过 slcand 创建的接口
```

代码会自动检测接口是否可用（`io/cboard.cpp` 中的 `check_socketcan_available()`），如果可用则使用 SocketCAN。

### 11.2 USB2CAN SDK 方式

如果 SocketCAN 不可用，代码会回退到 USB2CAN SDK（达妙科技）：

```yaml
# configs/example.yaml
usb2can_device: "/dev/ttyACM0"  # USB2CAN 设备路径
```

代码会按以下顺序尝试默认设备：
- `/dev/ttyACM0`（达妙科技 USB2CAN 常见路径）
- `/dev/USB2CAN0`
- `/dev/ttyACM1`

### 11.3 配置示例

完整配置示例（`configs/example.yaml` 片段）：

```yaml
#####-----cboard参数-----#####
quaternion_canid: 0x01      # 接收 IMU 四元数的 CAN ID
bullet_speed_canid: 0x101   # 接收弹速与模式信息的 CAN ID
send_canid: 0xff            # 下发控制命令的 CAN ID
can_interface: "can0"       # SocketCAN 接口名（优先）
# usb2can_device: "/dev/ttyACM0"  # USB2CAN 设备路径（备选）
```

### 11.4 代码中的使用

代码会自动选择可用的接口（`io/cboard.cpp`）：

```cpp
// 优先尝试 SocketCAN
if (!can_interface.empty() && check_socketcan_available(can_interface)) {
    socketcan_ = std::make_unique<SocketCAN>(...);
}

// 如果失败，回退到 USB2CAN SDK
if (!socketcan_ && check_usb2can_available(usb2can_device)) {
    usb2can_ = std::make_unique<USB2CAN>(...);
}
```

## 12. SocketCAN vs USB2CAN SDK 对比

| 特性 | SocketCAN（slcand） | USB2CAN SDK |
|------|---------------------|-------------|
| **兼容性** | 标准 Linux SocketCAN API，兼容性好 | 厂商特定，需要 SDK |
| **工具支持** | 可使用 `can-utils`（candump、cansend 等） | 需要厂商工具或 SDK |
| **调试便利性** | 高（标准工具链） | 中（依赖 SDK） |
| **性能** | 良好（内核驱动） | 取决于 SDK 实现 |
| **多设备支持** | 容易（can0, can1, ...） | 需要管理多个设备路径 |
| **自动重连** | 需要 systemd 服务管理 | SDK 可能内置重连机制 |
| **适用场景** | 生产环境、标准化部署 | 开发调试、厂商特定功能 |

**推荐**：优先使用 SocketCAN，便于调试和维护。

## 13. CAN 位速率（Bitrate）选择

常见 CAN 位速率及其应用场景：

| Bitrate | 应用场景 | 备注 |
|---------|---------|------|
| **125 kbps** | 低速应用、长距离 | 最大距离可达 500m |
| **250 kbps** | 中低速应用 | 平衡距离与速度 |
| **500 kbps** | **机器人控制（推荐）** | 本项目常用，平衡性能与稳定性 |
| **1 Mbps** | 高速应用、短距离 | 最大距离约 40m |

**设置示例**：

```bash
# 500 kbps（推荐用于机器人控制）
sudo ip link set can0 up type can bitrate 500000

# 1 Mbps（高速场景）
sudo ip link set can0 up type can bitrate 1000000

# 250 kbps（中速场景）
sudo ip link set can0 up type can bitrate 250000
```

**重要**：CAN 总线两端（发送端和接收端）的 bitrate **必须一致**，否则无法正常通信。

## 14. 回环模式测试

回环模式（Loopback）用于测试本地 CAN 接口，无需外部设备：

```bash
# 设置回环模式
sudo ip link set can0 up type can bitrate 500000 loopback on

# 在一个终端监听
candump can0

# 在另一个终端发送（会收到自己发送的数据）
cansend can0 123#11223344
```

**预期输出**（在 candump 终端）：
```
can0  123   [4]  11 22 33 44
```

**关闭回环模式**：
```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 500000 loopback off
```

## 15. 多设备场景

如果系统中有多个 USB2CAN 设备，需要为每个设备创建不同的 SocketCAN 接口：

```bash
# 第一个设备 -> can0
sudo slcand -S 115200 -o -c /dev/serial/by-id/usb-device-1 can0
sudo ip link set can0 up type can bitrate 500000

# 第二个设备 -> can1
sudo slcand -S 115200 -o -c /dev/serial/by-id/usb-device-2 can1
sudo ip link set can1 up type can bitrate 500000
```

**systemd 服务示例**（多设备）：

```ini
# /etc/systemd/system/slcand-usb2can-can0.service
[Unit]
Description=slcand for USB2CAN device 1 (can0)
After=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/slcand -S 115200 -o -c /dev/serial/by-id/usb-device-1 can0
ExecStartPost=/sbin/ip link set can0 up type can bitrate 500000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/slcand-usb2can-can1.service
[Unit]
Description=slcand for USB2CAN device 2 (can1)
After=network.target

[Service]
Type=simple
ExecStart=/usr/sbin/slcand -S 115200 -o -c /dev/serial/by-id/usb-device-2 can1
ExecStartPost=/sbin/ip link set can1 up type can bitrate 500000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## 16. 监控与统计

### 16.1 查看接口统计信息

```bash
# 查看详细统计（包括错误计数）
ip -s -s link show can0
```

**输出示例**：
```
3: can0: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UNKNOWN mode DEFAULT group default qlen 10
    link/can  promiscuity 0
    can <TRIPLE-SAMPLING> state ERROR-ACTIVE (berr-counter tx 0 rx 0) restart-ms 0
          bitrate 500000 sample-point 0.875
          tq 125 prop-seg 6 phase-seg1 7 phase-seg2 2 sjw 1
          slcand: ttyUSB0 at 115200
    RX: bytes  packets  errors  dropped overrun mcast
         1024      128       0        0       0     0
    TX: bytes  packets  errors  dropped carrier collsns
          512       64       0        0       0       0
```

**关键指标**：
- `errors`：错误计数（应为 0）
- `dropped`：丢包计数（应为 0）
- `berr-counter tx/rx`：总线错误计数（应为 0）

### 16.2 实时监控

```bash
# 监控所有 CAN 接口
watch -n 1 'ip -s link show can0'

# 或使用 cansniffer（需要安装 cansniffer）
cansniffer -c can0
```

### 16.3 日志记录

使用 `candump` 记录日志以便后续分析：

```bash
# 记录到文件（带时间戳）
candump -L -t z can0 > can0_$(date +%Y%m%d_%H%M%S).log

# 回放日志
canplayer -I can0_20250101_120000.log can0
```

## 17. 性能调优建议

### 17.1 缓冲区设置

增加 SocketCAN 接收缓冲区大小（减少丢包）：

```bash
# 设置接收队列长度（默认 10）
sudo ip link set can0 type can txqueuelen 1000

# 查看当前设置
ip link show can0
```

### 17.2 实时优先级（可选）

如果对实时性要求高，可以设置进程优先级：

```bash
# 使用 chrt 设置实时优先级
sudo chrt -f 50 candump can0
```

### 17.3 串口波特率优化

根据设备能力选择合适的串口波特率：

```bash
# 高速设备可使用 921600
sudo slcand -S 921600 -o -c /dev/ttyUSB0 can0

# 标准设备使用 115200
sudo slcand -S 115200 -o -c /dev/ttyUSB0 can0
```

**注意**：串口波特率不影响 CAN 总线 bitrate，但影响 USB2CAN 设备与主机之间的数据传输速度。

## 18. 常见错误与解决方案

### 18.1 "SIOCGIFINDEX: No such device"

**错误**：
```
Error getting interface index!
```

**原因**：接口不存在或未创建。

**解决**：
```bash
# 检查接口是否存在
ip link show can0

# 如果不存在，先创建
sudo slcand -S 115200 -o -c /dev/ttyUSB0 can0
sudo ip link set can0 up type can bitrate 500000
```

### 18.2 "Permission denied"

**错误**：
```
open: Permission denied
```

**原因**：用户没有访问串口设备的权限。

**解决**：
```bash
# 将用户添加到 dialout 组
sudo usermod -a -G dialout $USER

# 重新登录或使用 newgrp
newgrp dialout
```

### 18.3 "Device or resource busy"

**错误**：
```
slcand: ttyUSB0: Device or resource busy
```

**原因**：设备已被其他进程占用（可能是之前的 slcand 进程）。

**解决**：
```bash
# 查找并终止占用进程
ps aux | grep slcand
sudo pkill slcand

# 或查找占用 ttyUSB0 的进程
lsof /dev/ttyUSB0
sudo kill <PID>
```

### 18.4 "No buffer space available"

**错误**：
```
send: No buffer space available
```

**原因**：发送缓冲区满，可能是接收端未处理或 bitrate 不匹配。

**解决**：
```bash
# 增加发送队列长度
sudo ip link set can0 type can txqueuelen 1000

# 检查接收端是否正常工作
candump can0

# 确认 bitrate 一致
ip -details link show can0
```

### 18.5 "Network is down"

**错误**：
```
Network is down
```

**原因**：接口未启动。

**解决**：
```bash
# 启动接口
sudo ip link set can0 up type can bitrate 500000

# 检查状态
ip link show can0
```

### 18.6 无法收到数据

**排查步骤**：

1. **检查接口状态**：
   ```bash
   ip -details link show can0
   ```
   确认状态为 `UP` 且无错误。

2. **检查 bitrate 是否一致**：
   ```bash
   ip -details link show can0 | grep bitrate
   ```
   确认发送端和接收端 bitrate 相同。

3. **检查终端电阻**：
   CAN 总线两端需要各有一个 120Ω 终端电阻。

4. **回环测试**：
   ```bash
   sudo ip link set can0 type can loopback on
   candump can0 &
   cansend can0 123#11223344
   ```
   如果回环测试成功，问题在外部连接。

5. **检查错误计数**：
   ```bash
   ip -s link show can0 | grep errors
   ```
   如果错误计数持续增加，检查硬件连接和 bitrate。

### 18.7 slcand 进程立即退出

**排查**：

```bash
# 使用详细模式查看错误
sudo slcand -v -S 115200 -o -c /dev/ttyUSB0 can0

# 检查 systemd 日志
sudo journalctl -u slcand-usb2can.service -f

# 使用 strace 追踪系统调用
sudo strace -e trace=open,ioctl,write,read slcand -S 115200 -o -c /dev/ttyUSB0 can0
```

**常见原因**：
- 串口波特率不匹配（尝试 115200、921600 等）
- 设备不支持 SLCAN 协议（需要使用厂商 SDK）
- 权限不足

## 附：快速命令小结

```bash
# ========== 安装工具 ==========
sudo apt install can-utils usbutils

# ========== 设备识别 ==========
lsusb                                    # 查看 USB 设备
dmesg | tail -n 50                      # 查看内核日志
ls -l /dev/serial/by-id/*               # 查看串口设备

# ========== 绑定 & 启动 ==========
sudo slcand -S 115200 -o -c /dev/serial/by-id/your-usb2can-id can0
sudo ip link set can0 up type can bitrate 500000

# ========== 检查状态 ==========
ip -details link show can0              # 查看接口详情
ip -s link show can0                    # 查看统计信息

# ========== 调试 ==========
candump can0                            # 监听 CAN 数据
cansend can0 123#11223344               # 发送 CAN 数据
candump -L can0 > can0.log              # 记录日志

# ========== 回环测试 ==========
sudo ip link set can0 type can loopback on
candump can0 &
cansend can0 123#11223344

# ========== 停止 ==========
sudo ip link set can0 down              # 停止接口
sudo pkill slcand                       # 终止 slcand
sudo ip link delete can0                # 删除接口（可选）

# ========== systemd 管理 ==========
sudo systemctl enable --now slcand-usb2can.service
sudo systemctl status slcand-usb2can.service
sudo journalctl -u slcand-usb2can.service -f
```

---

## 参考资源

- [Linux SocketCAN 文档](https://www.kernel.org/doc/html/latest/networking/can.html)
- [can-utils 工具文档](https://github.com/linux-can/can-utils)
- [slcand man 手册](https://manpages.debian.org/testing/can-utils/slcand.8.en.html)
