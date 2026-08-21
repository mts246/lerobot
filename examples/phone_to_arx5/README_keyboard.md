# ARX X5 键盘笛卡尔遥操录制

`record_keyboard.py` 保留原 `record.py` 的 X5 SDK、相机观测、LeRobot 分集录制、
视频编码和数据集保存流程，只把手机 6DoF 输入替换成键盘末端 jog 输入。

## 启动

先确认 `can1` 已启动、机械臂周围没有人员或障碍物，并让急停/断电装置触手可及：

```bash
cd /home/user/x5/lerobot
conda run --no-capture-output -n lerobot \
  python examples/phone_to_arx5/record_keyboard.py
```

相机、集数、时长和输出目录参数与原脚本一致，例如：

```bash
conda run --no-capture-output -n lerobot \
  python examples/phone_to_arx5/record_keyboard.py \
  --can-interface can1 \
  --left-camera /dev/video10 \
  --right-camera /dev/video16 \
  --wrist-camera /dev/video4 \
  --fps 30 \
  --num-episodes 3 \
  --episode-time-s 30 \
  --reset-time-s 10 \
  --repo-id local/x5_keyboard_demo \
  --root /home/user/x5/outputs/x5_keyboard_demo
```

不用相机时传入空字符串：

```bash
conda run --no-capture-output -n lerobot \
  python examples/phone_to_arx5/record_keyboard.py \
  --left-camera '' --right-camera '' --wrist-camera ''
```

## 按键

按住运动键时持续运动，同时按下相反方向会相互抵消；松开全部末端运动键后立即保持当前位置。

| 功能 | 正方向 | 负方向 |
|---|---:|---:|
| X 平移 | `W` | `S` |
| Y 平移 | `A` | `D` |
| Z 平移 | `R` | `F` |
| Roll（翻转） | `L` | `J` |
| Pitch（俯仰） | `K` | `I` |
| Yaw | `U` | `O` |
| 夹爪 | `X` 打开 | `Z` 闭合 |

录制控制：

- `H`：回零；若正在录制，会丢弃当前 episode，然后在 reset 阶段回零。
- `Right` 或 `N`：提前结束并保存当前 episode。
- `Left` 或 `M`：丢弃并重新录制当前 episode。
- `Esc` 或 `Q`：停止录制并安全退出。

## 速度和工作空间限制

默认行为与手机遥操项目的 jog 模式一致：线速度上限 `0.18 m/s`、角速度上限
`45 deg/s`，XYZ 相对启动/回零位置限制为 `±(0.30, 0.30, 0.25) m`。可调低：

```bash
conda run --no-capture-output -n lerobot \
  python examples/phone_to_arx5/record_keyboard.py \
  --linear-speed 0.08 \
  --angular-speed-deg 20 \
  --session-limit 0.15 0.15 0.12
```

桌面 X11/Windows/macOS 环境使用真实的按下/松开事件。Wayland 或 SSH 下会退化为
终端短脉冲模式，按住键时依赖系统按键重复；要获得最可靠的“松开即停”，优先在 X11
桌面终端中运行。

## 退出保护

正常录制结束、`Esc/Q`、`Ctrl+C`、`SIGTERM`、`SIGHUP` 和可被 Python 捕获的异常都会
先停止控制，并按以下顺序清理机械臂：

1. `go_home(duration=0.5, wait=True)` 回零；
2. 无论回零是否成功，都继续执行 `damp(2.0)` 阻尼保护；
3. 如果阻尼指令失败，退回 `protect_mode()`；
4. 最后关闭 ARX SDK，再编码/整理数据集视频。

第一次退出信号触发清理后，后续退出信号会被暂时忽略，避免连续按 `Ctrl+C` 中断阻尼
过程。`SIGKILL`（例如 `kill -9`）、整机断电、内核故障或 SDK 原生进程崩溃无法由
Python 清理逻辑捕获，因此物理急停和可直接切断机械臂电源的装置仍然不可替代。
