# RealSense D435 Data Capture Tool

基于 Intel RealSense D435 的桌面俯视场景采集工具，提供 PyQt5 图形界面，支持三路相机（左/中/右）同时采集、RGB 实时预览与批量 metadata 导出。

## 1. 安装依赖

建议 Python 3.10+。

```bash
cd realsense_capture_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

说明：
- macOS 会安装 `pyrealsense2-macosx==2.54.2`（该版本在本项目中更稳定）
- Linux/Windows 会安装 `pyrealsense2`

## 2. 运行方式

```bash
cd realsense_capture_tool
python main.py
```

## 3. 工程目录说明

```text
realsense_capture_tool/
├── main.py
├── camera/
│   ├── __init__.py
│   └── realsense_camera.py
├── gui/
│   ├── __init__.py
│   └── main_window.py
├── processing/
│   ├── __init__.py
│   └── normals.py
├── utils/
│   ├── __init__.py
│   ├── metadata.py
│   ├── roi.py
│   └── saver.py
├── requirements.txt
└── README.md
```

## 4. 保存目录结构

默认保存到 `realsense_capture_tool/dataset_root/`，也可在 GUI 中手动切换。

```text
dataset_root/
├── camera_intrinsics.json
├── rgb/
│   ├── 000001_light1_left_rgb.png
│   ├── 000001_light1_middle_rgb.png
│   ├── 000001_light1_right_rgb.png
├── depth/
│   ├── 000001_light1_left_depth.png
│   ├── 000001_light1_middle_depth.png
│   ├── 000001_light1_right_depth.png
├── depth_vis/
│   ├── 000001_light1_left_depth_vis.png
│   ├── 000001_light1_middle_depth_vis.png
│   ├── 000001_light1_right_depth_vis.png
├── normals/
│   ├── 000001_light1_left_normals.png
│   ├── 000001_light1_middle_normals.png
│   ├── 000001_light1_right_normals.png
├── metadata.json
└── index.csv
```

## 5. metadata 导出机制

- 采集过程中，样本索引只保存在内存中（会话级列表）。
- 点击 `导出 metadata` 后统一写出：
  - `metadata.json`
  - `index.csv`
- 如果没有新变更，按钮会提示无需导出。
- 关闭程序时若存在未导出变更，会弹窗提醒是否导出。

每条记录包含：
- `sample_id`, `sample_name`, `light_label`
- `left_* / middle_* / right_*` 三路各自的 `rgb/depth/depth_vis/normals` 路径与 `width/height`
- `roi_enabled`, `roi_x`, `roi_y`, `roi_w`, `roi_h`
- `depth_aligned_to_color`, `intrinsics_file`, `saved_in_session`

## 6. 主要功能

- RealSense color + depth 同步采集
- 明确使用 `align(rs.stream.color)` 将 depth 对齐到 color
- 三路相机（左/中/右）并行采集
- 三路 RGB 实时预览（Depth/Normals 在保存时处理并写盘，减轻实时负载）
- ROI 设置、清除、预览叠加与保存裁剪
- 采集并保存当前三路帧（RGB/Depth16位/DepthVis/Normals）
- 删除上一张（含文件删除 + 内存 metadata 删除）
- 光照标签按钮组（默认 `light1/light2/light3`，支持动态添加）
- 分辨率和 FPS 可调
- 左/中/右设备序列号绑定与刷新
- 快捷键：
  - `S`: 采集并保存
  - `Backspace` 或 `Ctrl+Z`: 删除上一张

## 7. RealSense 使用注意事项

- 请确保 D435 连接在 USB 3.x 端口。
- 若未检测到设备：
  - 检查线缆、供电、权限（Linux 下可能需要 udev 规则）
  - 先使用 RealSense Viewer 验证设备是否可用
- 高分辨率 + 实时 normals 会增加 CPU 负载，可适当降低分辨率/FPS。

## 8. Depth PNG 保存说明

- `depth/*.png` 保存的是原始深度值（`uint16`）而非伪彩图。
- `depth_vis/*.png` 是便于人工查看的伪彩图，不用于精确计算。
- 深度单位转换依赖 `camera_intrinsics.json` 中的 `depth_scale`。

## 9. 常见操作流程

1. 启动程序并选择保存目录。
2. 选择分辨率/FPS，并绑定左/中/右三路设备序列号。
3. 选择光照标签并启动相机。
4. （可选）设置 ROI（ROI 会统一作用于三路保存）。
5. 点击 `采集并保存`（或按 `S`）采集当前三路。
6. 需要回退时点击 `删除上一张`。
7. 一批采集完成后点击 `导出 metadata`。

## 10. macOS 段错误排查

如果 `python main.py` 出现 `segmentation fault`：

- 当前项目已将设备探测改为“子进程探测”，可避免启动 GUI 时直接崩溃。
- 若启动相机提示 `RealSense 后端初始化失败`，通常是本机 `pyrealsense2/librealsense` 运行时不兼容。
- 推荐做法：
  - 使用 Python 3.10 新虚拟环境安装 `pyrealsense2`
  - 避免 Conda 与系统库混装
  - 先用 RealSense Viewer 验证设备
  - 采集任务优先在 Linux/Windows 执行（Intel 官方支持更稳定）
