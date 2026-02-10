# HIL-SERL 仿真环境运行指南（键盘控制）

本指南说明如何在使用 **仿真环境** 和 **键盘控制** 的情况下运行 LeRobot 的 HIL-SERL（人机协作强化学习）。

## 一、环境准备

### 1. 安装 HIL-SERL 依赖

需要安装 `gym_hil` 仿真包：

```bash
cd /home/zhang/robot/lerobot
pip install -e ".[hilserl]"
```

### 2. 硬件要求

- **NVIDIA GPU**（仿真基于 MuJoCo）
- **键盘**（用于控制机器人）

---

## 二、仿真任务说明

`gym_hil` 提供基于 MuJoCo 的 Franka Panda 机械臂仿真环境，支持以下任务：

| 任务名称 | 说明 |
|---------|------|
| `PandaPickCubeBase-v0` | 基础环境，无遥控 |
| `PandaPickCubeGamepad-v0` | 支持手柄控制 |
| `PandaPickCubeKeyboard-v0` | 支持键盘控制 |

使用键盘时请选择 **`PandaPickCubeKeyboard-v0`**。

---

## 三、运行方式

### 方式一：仅体验仿真（键盘控制）

先用键盘控制机械臂抓取立方体，不做数据采集或训练：

```bash
python -m lerobot.rl.gym_manipulator \
  --config_path examples/tutorial/rl/gym_hil_keyboard_config.json
```

此时 `mode` 为 `null`，仿真会启动，你可以用键盘控制机械臂。

### 方式二：录制演示数据

如需录制键盘操控的演示数据，在配置中设置 `mode: "record"` 并指定 `dataset`，然后运行：

```bash
# 先修改 gym_hil_keyboard_config.json 中的 mode 为 "record"
# 并设置 dataset.repo_id（例如 "你的用户名/franka_sim_keyboard"）

python -m lerobot.rl.gym_manipulator \
  --config_path examples/tutorial/rl/gym_hil_keyboard_config.json
```

> **注意**：录制模式下，需要先在 Hugging Face 创建对应的 dataset 仓库，或使用本地路径。

### 方式三：完整 HIL-SERL 训练（Actor + Learner）

若要在线训练策略，需要同时运行 **Learner** 和 **Actor** 两个进程。

**1. 下载示例训练配置：**

```bash
# 下载官方训练配置
wget -O examples/tutorial/rl/train_config_keyboard.json \
  "https://huggingface.co/datasets/lerobot/config_examples/resolve/main/rl/gym_hil/train_config.json"
```

**2. 修改配置为键盘环境：**

编辑 `train_config_keyboard.json`，将以下字段改为键盘相关：

```json
"env": {
  "task": "PandaPickCubeKeyboard-v0",
  "processor": {
    "control_mode": "keyboard",
    ...
  }
}
```

同时设置 `output_dir`，例如：

```json
"output_dir": "outputs/hil_serl_keyboard_sim"
```

**3. 先启动 Learner：**

```bash
python -m lerobot.rl.learner --config_path examples/tutorial/rl/train_config_keyboard.json
```

**4. 在另一个终端启动 Actor：**

```bash
python -m lerobot.rl.actor --config_path examples/tutorial/rl/train_config_keyboard.json
```

**训练流程说明：**

- Learner 负责训练策略并推送参数
- Actor 在仿真中执行策略，并在需要时通过键盘介入
- 按住键盘控制键时，你的操作会覆盖策略动作，形成 human-in-the-loop 干预

---

## 四、键盘控制说明

`PandaPickCubeKeyboard-v0` 的典型按键（具体以 `gym_hil` 文档为准）：

- **方向键**：控制末端执行器在 x/y 平面移动
- **Shift**：控制 z 轴
- **其他键**：夹爪开合、重置等

仿真窗口出现后，请关注终端或 `gym_hil` 的文档以确认完整按键列表。

---

## 五、常见问题

### 1. 提示找不到 `gym_hil`

请确认已安装 HIL-SERL 依赖：

```bash
pip install -e ".[hilserl]"
```

### 2. 没有 GPU / CUDA

若只有 CPU，可将配置中的 `device` 改为 `"cpu"`，但仿真会较慢。

### 3. 与 `hilserl_example.py` 的区别

`examples/tutorial/rl/hilserl_example.py` 针对 **真实 SO100 机械臂**，依赖串口和遥操作设备。  
仿真环境应使用：

- `gym_manipulator`（体验、录制）
- `actor` + `learner`（训练）

而不是直接运行 `hilserl_example.py`。

---

## 六、配置文件说明

`gym_hil_keyboard_config.json` 主要字段：

| 字段 | 说明 |
|------|------|
| `env.task` | `PandaPickCubeKeyboard-v0` 表示键盘控制 |
| `env.processor.control_mode` | `"keyboard"` |
| `env.fps` | 控制频率（如 10 Hz） |
| `mode` | `null`=仅体验，`record`=录制 |
| `dataset.repo_id` | 录制时的数据集仓库 ID |

---

## 七、参考文档

- [HIL-SERL 仿真文档](https://huggingface.co/docs/lerobot/hilserl_sim)
- [HIL-SERL 主文档](https://huggingface.co/docs/lerobot/hilserl)
- [gym_hil 配置示例](https://huggingface.co/datasets/lerobot/config_examples/tree/main/rl/gym_hil)
