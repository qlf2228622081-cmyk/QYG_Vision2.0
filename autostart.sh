#!/usr/bin/env bash

# 记录启动时间
START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "[autostart] 服务启动时间: ${START_TIME}" >&2
echo "[autostart] 服务启动时间: ${START_TIME}" | logger -t rm_vision

sleep 5

# 切换到当前工程目录（请确认路径是否与实际一致）
cd /home/rm/rm_vision_2025 || exit 1

# 记录进入目录后的时间
AFTER_SLEEP_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "[autostart] 等待完成，开始启动 watchdog: ${AFTER_SLEEP_TIME}" >&2
echo "[autostart] 等待完成，开始启动 watchdog: ${AFTER_SLEEP_TIME}" | logger -t rm_vision

# 使用 screen 后台运行 watchdog.sh，并把标准输出写入日志
SCREEN_LOG="logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog"
echo "[autostart] Screen 日志文件: ${SCREEN_LOG}" >&2
echo "[autostart] Screen 日志文件: ${SCREEN_LOG}" | logger -t rm_vision

# 启动 screen 并运行 watchdog.sh
screen \
    -L \
    -Logfile "${SCREEN_LOG}" \
    -d -m \
    bash -c "./watchdog.sh"

# 检查 screen 是否成功启动
sleep 2
SCREEN_OUTPUT=$(screen -ls 2>&1)
if echo "${SCREEN_OUTPUT}" | grep -q "Detached"; then
    SCREEN_COUNT=$(echo "${SCREEN_OUTPUT}" | grep -c "Detached" || echo "0")
    echo "[autostart] Screen 会话已启动，当前有 ${SCREEN_COUNT} 个会话" >&2
    echo "[autostart] Screen 会话已启动，当前有 ${SCREEN_COUNT} 个会话" | logger -t rm_vision
else
    echo "[autostart] 警告：无法确认 Screen 会话状态，但继续运行..." >&2
    echo "[autostart] 警告：无法确认 Screen 会话状态，但继续运行..." | logger -t rm_vision
fi

# 记录启动完成时间
END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
echo "[autostart] watchdog 启动完成时间: ${END_TIME}" >&2
echo "[autostart] watchdog 启动完成时间: ${END_TIME}" | logger -t rm_vision

# 保持脚本运行，监控 screen 会话
while true; do
    sleep 30
    # 检查是否有 watchdog 相关的 screen 会话在运行
    WATCHDOG_SCREENS=$(screen -ls | grep -c "Detached" || echo "0")
    if [[ "${WATCHDOG_SCREENS}" -eq 0 ]]; then
        echo "[autostart] 没有 Screen 会话在运行，重新启动..." >&2
        echo "[autostart] 没有 Screen 会话在运行，重新启动..." | logger -t rm_vision
        
        SCREEN_LOG="logs/$(date "+%Y-%m-%d_%H-%M-%S").screenlog"
        screen \
            -L \
            -Logfile "${SCREEN_LOG}" \
            -d -m \
            bash -c "./watchdog.sh"
        
        sleep 2
        NEW_SCREEN_COUNT=$(screen -ls | grep -c "Detached" || echo "0")
        if [[ "${NEW_SCREEN_COUNT}" -gt 0 ]]; then
            echo "[autostart] Screen 会话重新启动成功" >&2
            echo "[autostart] Screen 会话重新启动成功" | logger -t rm_vision
        else
            echo "[autostart] 警告：无法确认 Screen 会话是否重新启动" >&2
            echo "[autostart] 警告：无法确认 Screen 会话是否重新启动" | logger -t rm_vision
        fi
    fi
done
