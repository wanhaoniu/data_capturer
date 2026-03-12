# Hand-Eye Calibration GUI (Ubuntu/Linux)

一个可运行的 Python GUI 工具，用于机器人手眼标定，支持 Intel RealSense D435 与棋盘格标定板。

- 支持模式: Eye-in-Hand / Eye-to-Hand
- 支持算法: Tsai / Park / Daniilidis
- 支持棋盘格参数自定义: 列数、行数、方块尺寸
- 支持接入 UR 机器人，检索常用通信端口并通过官方 RTDE Python Client Library 读取 `actual_TCP_pose`
- 支持工程保存/加载与结果导出 (JSON / YAML / TXT)
- 不依赖 ROS

## 项目结构

```text
./
├── README.md
├── requirements.txt
├── __init__.py
├── main.py
├── ui
│   ├── __init__.py
│   └── main_window.py
├── camera
│   ├── __init__.py
│   └── realsense_camera.py
├── robot
│   ├── __init__.py
│   └── ur_rtde_client.py
├── calibration
│   ├── __init__.py
│   ├── chessboard_detector.py
│   ├── handeye_solver.py
│   └── transforms.py
├── data
│   ├── __init__.py
│   └── sample_manager.py
└── utils
    ├── __init__.py
    └── io_utils.py
```

## 坐标系约定

- `T_base_tool`: 机器人基座到工具坐标系。可以是法兰/末端执行器，也可以是 UR 当前 TCP，但整个采样过程必须保持同一个工具定义
- `T_cam_target`: 相机到棋盘格目标（由 `solvePnP`）

### Eye-in-Hand

- 相机安装在末端
- 求解输出: `T_tool_camera`
- 一致性评估: `T_base_target = T_base_tool * T_tool_camera * T_cam_target`

### Eye-to-Hand

- 相机固定在环境中
- 假设棋盘格固定在末端执行器（常见外参标定方式）
- 求解输出: `T_base_camera`
- 一致性评估: `T_tool_target = T_tool_base * T_base_camera * T_cam_target`

## 工程文件格式

保存工程目录后结构为:

```text
project/
├── images/
├── samples.json
├── config.json
├── intrinsics.json
└── result.json
```

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果 `pyrealsense2` 安装失败，请先安装 Intel RealSense SDK（`librealsense2`）并确认 Python 绑定可用。

UR 接口使用 Universal Robots 官方 `RTDE_Python_Client_Library`。如果你的环境无法直接访问 GitHub，也可以手动安装官方仓库后再运行。

## 运行

```bash
python3 main.py
```

## GUI 使用流程

1. 打开相机
2. 设置棋盘格参数（列、行、方块尺寸）
3. 在 `UR 机器人接口` 中填写机器人 IP，先点 `检索端口`，确认 `30004` 可用后连接
4. 连接成功后，程序会通过 RTDE 订阅 `actual_TCP_pose = (x, y, z, rx, ry, rz)`，并自动换算到界面显示的 `xyz + roll pitch yaw`
5. 如果不接 UR，也可以继续手动输入当前机器人位姿
6. 检测棋盘格（检测成功后会绘制角点和坐标轴）
7. 采集样本。若 UR 已连接，采样前会自动刷新一次 TCP 位姿，避免使用旧值
8. 开始标定（选择模式与方法）
9. 查看误差并保存工程/导出结果

## UR 使用说明

- 连接端口使用 RTDE，默认 `30004`
- 检索端口会扫描 UR 常用通信端口: `29999, 30001, 30002, 30003, 30004, 30011, 30012, 30013, 30020`
- 读取字段使用官方 RTDE 输出字段 `actual_TCP_pose`
- `actual_TCP_pose` 的姿态部分是旋转向量 `(rx, ry, rz)`，不是欧拉角，程序内部会先转成 4x4 变换矩阵，再显示为 `roll/pitch/yaw`
- 如果没有发现 `30004`，请检查 PolyScope 的 RTDE 服务和防火墙/入站访问策略

## 结果导出

- JSON: 完整标定结果与误差
- YAML: 与 JSON 同内容
- TXT: 4x4 变换矩阵（主结果）
