from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from camera.realsense_camera import CameraConfig, RealSenseCamera
from processing.normals import NormalsEstimator
from utils.metadata import SessionMetadataManager
from utils.roi import ROI, select_roi_with_opencv
from utils.saver import DatasetSaver

CAMERA_KEYS = ("left", "middle", "right")
CAMERA_TEXT = {
    "left": "左",
    "middle": "中",
    "right": "右",
}


class CameraWorker(QThread):
    frame_ready = pyqtSignal(str, object, object)  # camera_key, color_bgr, depth_u16
    camera_started = pyqtSignal(str, dict)  # camera_key, intrinsics
    camera_stopped = pyqtSignal(str)  # camera_key
    error_signal = pyqtSignal(str, str)  # camera_key, message

    def __init__(self, camera_key: str, serial_number: str, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.camera_key = camera_key
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self._stop_event = Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        camera = RealSenseCamera()
        try:
            camera.start(
                CameraConfig(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    serial_number=self.serial_number,
                )
            )
            intrinsics = camera.get_intrinsics()
            self.camera_started.emit(self.camera_key, intrinsics)

            while not self._stop_event.is_set():
                try:
                    color_bgr, depth_u16 = camera.get_aligned_frames(timeout_ms=1000)
                except Exception as exc:
                    self.error_signal.emit(self.camera_key, str(exc))
                    continue

                self.frame_ready.emit(self.camera_key, color_bgr.copy(), depth_u16.copy())

        except Exception as exc:
            self.error_signal.emit(self.camera_key, str(exc))
        finally:
            camera.stop()
            self.camera_stopped.emit(self.camera_key)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("RealSense D435 三路采集工具（左 / 中 / 右）")
        self.resize(1700, 980)

        default_root = str((Path.cwd() / "dataset_root").resolve())
        self.saver = DatasetSaver(default_root)
        self.metadata_manager = SessionMetadataManager()

        self.workers: Dict[str, CameraWorker] = {}
        self.device_catalog: list[Dict] = []
        self.view_to_source: Dict[str, str] = {}
        self.source_to_views: Dict[str, list[str]] = {}
        self.started_sources: Dict[str, Dict] = {}

        self.camera_intrinsics_by_cam: Dict[str, Dict] = {}
        self.normals_estimators: Dict[str, NormalsEstimator] = {}

        self.latest_color_bgr: Dict[str, np.ndarray] = {}
        self.latest_depth_u16: Dict[str, np.ndarray] = {}

        self.current_roi: Optional[ROI] = None

        self.light_button_group = QButtonGroup(self)
        self.light_button_group.setExclusive(True)
        self.light_buttons: Dict[str, QRadioButton] = {}

        self.preview_views: Dict[str, Dict[str, QWidget]] = {}
        self.device_combos: Dict[str, QComboBox] = {}

        self._build_ui()
        self._apply_styles()
        self._bind_signals()
        self._setup_shortcuts()
        self._set_initial_status()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)

        preview_group = QGroupBox("预览区（左 / 中 / 右 RGB）")
        preview_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)

        hint = QLabel("实时仅显示 RGB。Depth 与 Normals 在保存时计算并写盘。")
        hint.setObjectName("PreviewHint")
        hint.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(hint)

        preview_splitter = QSplitter(Qt.Horizontal)
        preview_splitter.setChildrenCollapsible(False)
        preview_splitter.setHandleWidth(8)

        for key in CAMERA_KEYS:
            panel = self._make_camera_panel(f"{CAMERA_TEXT[key]}路相机")
            self.preview_views[key] = panel
            preview_splitter.addWidget(panel["container"])

        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setStretchFactor(2, 1)
        preview_layout.addWidget(preview_splitter)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_container.setMinimumWidth(430)

        controls_group = QGroupBox("控制按钮区")
        controls_layout = QGridLayout(controls_group)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(12)

        self.btn_start_camera = QPushButton("启动相机")
        self.btn_stop_camera = QPushButton("停止相机")
        self.btn_capture_save = QPushButton("采集并保存")
        self.btn_delete_last = QPushButton("删除上一张")
        self.btn_export_metadata = QPushButton("导出 metadata")
        self.btn_exit = QPushButton("退出程序")
        self.btn_exit.setObjectName("DangerButton")

        controls_layout.addWidget(self.btn_start_camera, 0, 0)
        controls_layout.addWidget(self.btn_stop_camera, 0, 1)
        controls_layout.addWidget(self.btn_capture_save, 0, 2)
        controls_layout.addWidget(self.btn_delete_last, 1, 0)
        controls_layout.addWidget(self.btn_export_metadata, 1, 1)
        controls_layout.addWidget(self.btn_exit, 1, 2)
        controls_layout.setColumnStretch(0, 1)
        controls_layout.setColumnStretch(1, 1)
        controls_layout.setColumnStretch(2, 1)

        params_group = QGroupBox("参数设置区")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(12)

        save_path_row = QHBoxLayout()
        self.edit_save_root = QLineEdit(str(self.saver.root_dir))
        self.edit_save_root.setReadOnly(True)
        self.btn_choose_root = QPushButton("选择保存目录")
        save_path_row.addWidget(self.edit_save_root)
        save_path_row.addWidget(self.btn_choose_root)
        params_layout.addLayout(save_path_row)

        resolution_row = QHBoxLayout()
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItems(["640x480", "848x480", "1280x720"])
        self.combo_resolution.setCurrentText("640x480")
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(6, 60)
        self.spin_fps.setValue(30)
        resolution_row.addWidget(QLabel("分辨率:"))
        resolution_row.addWidget(self.combo_resolution)
        resolution_row.addWidget(QLabel("FPS:"))
        resolution_row.addWidget(self.spin_fps)
        params_layout.addLayout(resolution_row)

        serial_group = QGroupBox("设备绑定（左 / 中 / 右）")
        serial_layout = QGridLayout(serial_group)
        serial_layout.setHorizontalSpacing(8)
        serial_layout.setVerticalSpacing(8)

        for idx, key in enumerate(CAMERA_KEYS):
            combo = QComboBox()
            combo.setObjectName(f"combo_{key}")
            self.device_combos[key] = combo
            serial_layout.addWidget(QLabel(f"{CAMERA_TEXT[key]}路设备:"), idx, 0)
            serial_layout.addWidget(combo, idx, 1)

        self.btn_refresh_devices = QPushButton("刷新设备列表")
        serial_layout.addWidget(self.btn_refresh_devices, 3, 1)
        params_layout.addWidget(serial_group)

        light_group = QGroupBox("光照标签（点选）")
        light_layout = QVBoxLayout(light_group)
        self.light_buttons_row = QHBoxLayout()

        for label in ["light1", "light2", "light3"]:
            self._add_light_label_button(label, checked=(label == "light1"))

        light_layout.addLayout(self.light_buttons_row)

        custom_light_row = QHBoxLayout()
        self.edit_custom_light = QLineEdit()
        self.edit_custom_light.setPlaceholderText("输入自定义光照标签")
        self.btn_add_light = QPushButton("添加标签")
        custom_light_row.addWidget(self.edit_custom_light)
        custom_light_row.addWidget(self.btn_add_light)
        light_layout.addLayout(custom_light_row)

        params_layout.addWidget(light_group)

        roi_row = QHBoxLayout()
        self.check_enable_roi = QCheckBox("启用 ROI")
        self.btn_set_roi = QPushButton("设置 ROI")
        self.btn_clear_roi = QPushButton("清除 ROI")
        roi_row.addWidget(self.check_enable_roi)
        roi_row.addWidget(self.btn_set_roi)
        roi_row.addWidget(self.btn_clear_roi)
        params_layout.addLayout(roi_row)

        status_group = QGroupBox("状态显示区")
        status_layout = QFormLayout(status_group)
        status_layout.setVerticalSpacing(14)
        status_layout.setHorizontalSpacing(20)
        status_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        status_layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        status_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignTop)
        status_layout.setFormAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.lbl_camera_status = self._make_value_label("未启动")
        self.lbl_resolution_status = self._make_value_label("640x480 @ 30 FPS")
        self.lbl_save_path_status = self._make_value_label(str(self.saver.root_dir))
        self.lbl_light_status = self._make_value_label(self.get_selected_light_label())
        self.lbl_sample_id_status = self._make_value_label(f"{self.saver.next_sample_id():06d}")
        self.lbl_last_save_status = self._make_value_label("暂无")
        self.lbl_last_delete_status = self._make_value_label("暂无")
        self.lbl_metadata_status = self._make_value_label("已导出")
        self.lbl_roi_status = self._make_value_label("未启用")

        status_layout.addRow("相机状态:", self.lbl_camera_status)
        status_layout.addRow("当前分辨率:", self.lbl_resolution_status)
        status_layout.addRow("当前保存路径:", self.lbl_save_path_status)
        status_layout.addRow("当前光照标签:", self.lbl_light_status)
        status_layout.addRow("当前样本编号:", self.lbl_sample_id_status)
        status_layout.addRow("最近一次保存结果:", self.lbl_last_save_status)
        status_layout.addRow("最近一次删除结果:", self.lbl_last_delete_status)
        status_layout.addRow("metadata 状态:", self.lbl_metadata_status)
        status_layout.addRow("ROI 状态:", self.lbl_roi_status)

        right_layout.addWidget(controls_group)
        right_layout.addWidget(params_group)
        right_layout.addWidget(status_group)
        right_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        right_scroll.setWidget(right_container)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.addWidget(preview_group)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([1200, 560])
        root_layout.addWidget(splitter)

        self.statusBar().showMessage("就绪")

    def _make_camera_panel(self, title: str) -> Dict[str, QWidget]:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        image_label = QLabel("No Frame")
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(280, 210)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        image_label.setStyleSheet(
            "border: 1px solid #a7bac8; border-radius: 10px; background: #0d1e2a; color: #95a9ba;"
        )

        layout.addWidget(title_label)
        layout.addWidget(image_label)

        return {"container": container, "image_label": image_label}

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef3f6; }
            QWidget {
                font-family: "PingFang SC", "Helvetica Neue", "Microsoft YaHei";
                font-size: 13px;
                color: #1f2d3d;
            }
            QGroupBox {
                border: 1px solid #c8d3dd;
                border-radius: 12px;
                margin-top: 12px;
                background: #ffffff;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #16324f;
                background: transparent;
            }
            QPushButton {
                background: #165d88;
                color: #ffffff;
                border: none;
                border-radius: 8px;
                padding: 7px 12px;
                min-height: 30px;
            }
            QPushButton:hover { background: #1b6d9d; }
            QPushButton:pressed { background: #124c70; }
            QPushButton:disabled { background: #b3c3cf; color: #e7edf2; }
            QPushButton#DangerButton { background: #b04542; }
            QPushButton#DangerButton:hover { background: #993a37; }
            QLineEdit, QComboBox, QSpinBox {
                background: #f8fbfd;
                border: 1px solid #b7c5d1;
                border-radius: 8px;
                padding: 6px;
                min-height: 30px;
            }
            QRadioButton, QCheckBox { spacing: 7px; padding: 2px; }
            QLabel#PreviewHint { color: #4f6273; font-size: 12px; }
            QLabel#StatusValue {
                color: #102a43;
                font-weight: 600;
                padding: 2px 0;
                line-height: 1.45;
            }
            QStatusBar { background: #dfe8ef; color: #1f2d3d; }
            """
        )

    def _make_value_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("StatusValue")
        lbl.setWordWrap(True)
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lbl.setMinimumHeight(24)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return lbl

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_all_previews()

    def _bind_signals(self) -> None:
        self.btn_start_camera.clicked.connect(self.start_camera)
        self.btn_stop_camera.clicked.connect(self.stop_camera)
        self.btn_capture_save.clicked.connect(self.capture_and_save)
        self.btn_delete_last.clicked.connect(self.delete_last_sample)
        self.btn_export_metadata.clicked.connect(self.export_metadata)
        self.btn_exit.clicked.connect(self.close)

        self.btn_choose_root.clicked.connect(self.choose_save_root)
        self.btn_add_light.clicked.connect(self.add_custom_light_label)
        self.btn_refresh_devices.clicked.connect(self.refresh_devices)

        self.combo_resolution.currentTextChanged.connect(self._on_resolution_or_fps_changed)
        self.spin_fps.valueChanged.connect(self._on_resolution_or_fps_changed)
        self.light_button_group.buttonClicked.connect(self._on_light_label_changed)

        self.check_enable_roi.toggled.connect(self._on_roi_toggled)
        self.btn_set_roi.clicked.connect(self.set_roi)
        self.btn_clear_roi.clicked.connect(self.clear_roi)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence("S"), self, activated=self.capture_and_save)
        QShortcut(QKeySequence("Backspace"), self, activated=self.delete_last_sample)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.delete_last_sample)

    def _set_initial_status(self) -> None:
        self.btn_stop_camera.setEnabled(False)
        self.btn_capture_save.setEnabled(False)
        self.btn_delete_last.setEnabled(False)

        self.lbl_save_path_status.setText(str(self.saver.root_dir))
        self.lbl_sample_id_status.setText(f"{self.saver.next_sample_id():06d}")

        self.refresh_devices()
        self._update_metadata_status()
        self._update_roi_status()

    def refresh_devices(self) -> None:
        probe = RealSenseCamera.probe_device_list_safe()
        if not probe.get("backend_ok"):
            self.lbl_camera_status.setText("设备探测失败")
            self.statusBar().showMessage(f"设备探测失败: {probe.get('error', '未知错误')}", 6000)
            return

        devices = probe.get("devices", [])
        filtered = []
        for item in devices:
            serial = str(item.get("serial_number", "")).strip()
            name = str(item.get("name", "RealSense")).strip() or "RealSense"
            if serial:
                filtered.append({"serial_number": serial, "name": name})

        self.device_catalog = filtered

        previous = {
            key: self.device_combos[key].currentData() if self.device_combos[key].count() > 0 else ""
            for key in CAMERA_KEYS
        }

        for key in CAMERA_KEYS:
            combo = self.device_combos[key]
            combo.blockSignals(True)
            combo.clear()
            if not filtered:
                combo.addItem("未检测到设备", "")
            else:
                for dev in filtered:
                    label = f"{dev['name']} | SN:{dev['serial_number']}"
                    combo.addItem(label, dev["serial_number"])

                prev_serial = previous.get(key, "")
                if prev_serial:
                    idx = combo.findData(prev_serial)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                elif combo.count() > 0:
                    default_idx = min(CAMERA_KEYS.index(key), combo.count() - 1)
                    combo.setCurrentIndex(default_idx)
            combo.blockSignals(False)

        self.lbl_camera_status.setText(f"检测到设备 {len(filtered)} 台")
        self.statusBar().showMessage(f"设备列表已刷新: {len(filtered)} 台")

    def _get_selected_serial_mapping(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for key in CAMERA_KEYS:
            mapping[key] = str(self.device_combos[key].currentData() or "").strip()
        return mapping

    def _on_resolution_or_fps_changed(self) -> None:
        self.lbl_resolution_status.setText(f"{self.combo_resolution.currentText()} @ {self.spin_fps.value()} FPS")

    def _on_light_label_changed(self) -> None:
        self.lbl_light_status.setText(self.get_selected_light_label())

    def _on_roi_toggled(self, _: bool) -> None:
        self._update_roi_status()
        self._refresh_all_previews()

    def _add_light_label_button(self, label: str, checked: bool = False) -> None:
        if label in self.light_buttons:
            self.light_buttons[label].setChecked(True)
            self.lbl_light_status.setText(self.get_selected_light_label())
            return

        btn = QRadioButton(label)
        btn.setChecked(checked)
        self.light_buttons[label] = btn
        self.light_button_group.addButton(btn)
        self.light_buttons_row.addWidget(btn)

    def add_custom_light_label(self) -> None:
        raw = self.edit_custom_light.text().strip()
        if not raw:
            QMessageBox.warning(self, "提示", "请输入标签名称。")
            return

        label = DatasetSaver.sanitize_light_label(raw)
        self._add_light_label_button(label, checked=True)
        self.edit_custom_light.clear()
        self.lbl_light_status.setText(self.get_selected_light_label())

    def get_selected_light_label(self) -> str:
        btn = self.light_button_group.checkedButton()
        return btn.text() if btn else "light1"

    def parse_resolution(self) -> Tuple[int, int]:
        text = self.combo_resolution.currentText()
        try:
            w_str, h_str = text.split("x")
            return int(w_str), int(h_str)
        except Exception:
            return 640, 480

    def start_camera(self) -> None:
        if self.workers:
            QMessageBox.information(self, "提示", "相机已在运行。")
            return

        serial_map = self._get_selected_serial_mapping()
        if any(not serial_map[k] for k in CAMERA_KEYS):
            QMessageBox.warning(self, "提示", "请为左/中/右三路都选择设备序列号。")
            return

        self.view_to_source = dict(serial_map)
        self.source_to_views = {}
        for view_key, serial in self.view_to_source.items():
            self.source_to_views.setdefault(serial, []).append(view_key)
        unique_serials = list(self.source_to_views.keys())

        width, height = self.parse_resolution()
        fps = self.spin_fps.value()

        for serial in unique_serials:
            start_probe = RealSenseCamera.probe_stream_start_safe(
                width=width,
                height=height,
                fps=fps,
                serial_number=serial,
                timeout_sec=8.0,
            )
            if not start_probe.get("ok"):
                err_text = start_probe.get("error", "未知错误")
                bound_views = ",".join(CAMERA_TEXT.get(v, v) for v in self.source_to_views.get(serial, []))
                QMessageBox.critical(
                    self,
                    "相机启动预检失败",
                    f"设备 SN:{serial}（绑定到 {bound_views}）无法安全启动，已阻止启动。\n详细信息: {err_text}\n\n"
                    "请检查 USB3 直连、供电、线缆及设备占用。",
                )
                self.statusBar().showMessage(f"设备 SN:{serial} 预检失败: {err_text}", 8000)
                return

        self.camera_intrinsics_by_cam.clear()
        self.normals_estimators.clear()
        self.latest_color_bgr.clear()
        self.latest_depth_u16.clear()
        self.started_sources.clear()

        for serial in unique_serials:
            worker = CameraWorker(
                camera_key=serial,
                serial_number=serial,
                width=width,
                height=height,
                fps=fps,
            )
            worker.frame_ready.connect(self._on_frame_ready)
            worker.camera_started.connect(self._on_camera_started)
            worker.camera_stopped.connect(self._on_camera_stopped)
            worker.error_signal.connect(self._on_worker_error)
            worker.start()
            self.workers[serial] = worker

        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        self.btn_capture_save.setEnabled(False)
        self.lbl_camera_status.setText(f"启动中: 0/{len(unique_serials)}（视图: 3）")
        self.statusBar().showMessage(
            f"相机启动中: 物理设备 {len(unique_serials)} 台，逻辑视图 3 路"
        )

    def stop_camera(self) -> None:
        if not self.workers:
            return

        for worker in self.workers.values():
            if worker.isRunning():
                worker.stop()

        for worker in self.workers.values():
            if worker.isRunning():
                worker.wait(3000)

        self.workers.clear()
        self.btn_start_camera.setEnabled(True)
        self.btn_stop_camera.setEnabled(False)
        self.btn_capture_save.setEnabled(False)
        self.lbl_camera_status.setText("已停止")
        self.statusBar().showMessage("相机已停止")

    def _on_camera_started(self, source_key: str, intrinsics: Dict) -> None:
        self.started_sources[source_key] = intrinsics

        bound_views = self.source_to_views.get(source_key, [])
        for view_key in bound_views:
            self.camera_intrinsics_by_cam[view_key] = intrinsics
            self.normals_estimators[view_key] = NormalsEstimator.from_intrinsics_dict(intrinsics)

        started = len(self.started_sources)
        total = max(len(self.source_to_views), 1)
        self.lbl_camera_status.setText(f"运行中: {started}/{total}（视图: 3）")

        if started == total:
            payload = {
                "camera_mapping": self._get_selected_serial_mapping(),
                "cameras": {k: self.camera_intrinsics_by_cam.get(k, {}) for k in CAMERA_KEYS},
            }
            try:
                intr_file = self.saver.save_intrinsics(payload)
                self.statusBar().showMessage(
                    f"相机已启动（物理设备 {total} 台，逻辑视图 3 路），内参已保存: {intr_file}"
                )
            except Exception as exc:
                QMessageBox.warning(self, "警告", str(exc))

            self.btn_capture_save.setEnabled(True)

    def _on_camera_stopped(self, source_key: str) -> None:
        if source_key in self.workers and not self.workers[source_key].isRunning():
            self.workers.pop(source_key, None)
        self.started_sources.pop(source_key, None)

        if not self.workers:
            self.btn_start_camera.setEnabled(True)
            self.btn_stop_camera.setEnabled(False)
            self.btn_capture_save.setEnabled(False)
            self.lbl_camera_status.setText("已停止")

    def _on_worker_error(self, source_key: str, msg: str) -> None:
        bound_views = ",".join(CAMERA_TEXT.get(v, v) for v in self.source_to_views.get(source_key, []))
        if bound_views:
            self.statusBar().showMessage(f"设备 SN:{source_key}（{bound_views}）: {msg}", 6000)
        else:
            self.statusBar().showMessage(f"设备 SN:{source_key}: {msg}", 6000)

    def _on_frame_ready(self, source_key: str, color_bgr: np.ndarray, depth_u16: np.ndarray) -> None:
        bound_views = self.source_to_views.get(source_key, [])
        if not bound_views:
            return

        for view_key in bound_views:
            self.latest_color_bgr[view_key] = color_bgr
            self.latest_depth_u16[view_key] = depth_u16
            self._refresh_preview(view_key)

    def _refresh_preview(self, camera_key: str) -> None:
        img = self.latest_color_bgr.get(camera_key)
        if img is None:
            return

        disp = img.copy()
        if self.check_enable_roi.isChecked() and self.current_roi is not None:
            roi = self.current_roi.clamp(disp.shape[:2])
            if roi.is_valid():
                cv2.rectangle(disp, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 2)

        label = self.preview_views[camera_key]["image_label"]
        self._set_image_to_label(label, disp)

    def _refresh_all_previews(self) -> None:
        for key in CAMERA_KEYS:
            self._refresh_preview(key)

    def _set_image_to_label(self, label: QLabel, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(pix)

    def choose_save_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择保存目录", str(self.saver.root_dir))
        if not selected:
            return

        if self.metadata_manager.record_count > 0:
            reply = QMessageBox.question(
                self,
                "确认切换",
                "切换保存目录将清空当前会话 metadata 索引。是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self.metadata_manager.clear()
            self.lbl_last_save_status.setText("已清空会话")
            self.lbl_last_delete_status.setText("已清空会话")

        try:
            self.saver.set_root_dir(selected)
        except Exception as exc:
            QMessageBox.critical(self, "错误", str(exc))
            return

        self.edit_save_root.setText(str(self.saver.root_dir))
        self.lbl_save_path_status.setText(str(self.saver.root_dir))
        self.lbl_sample_id_status.setText(f"{self.saver.next_sample_id():06d}")
        self._update_metadata_status()
        self.btn_delete_last.setEnabled(self.metadata_manager.record_count > 0)

        if len(self.camera_intrinsics_by_cam) == 3:
            payload = {
                "camera_mapping": self._get_selected_serial_mapping(),
                "cameras": {k: self.camera_intrinsics_by_cam[k] for k in CAMERA_KEYS},
            }
            try:
                self.saver.save_intrinsics(payload)
            except Exception as exc:
                QMessageBox.warning(self, "警告", f"切换目录后写入内参失败: {exc}")

        self.statusBar().showMessage(f"保存目录已切换: {self.saver.root_dir}")

    def set_roi(self) -> None:
        source = self.latest_color_bgr.get("middle")
        if source is None:
            for key in CAMERA_KEYS:
                source = self.latest_color_bgr.get(key)
                if source is not None:
                    break

        if source is None:
            QMessageBox.warning(self, "提示", "当前没有可用的 RGB 帧，无法设置 ROI。")
            return

        roi = select_roi_with_opencv(source, window_name="Select ROI (Enter确认 / C取消)")
        if roi is None:
            self.statusBar().showMessage("未设置 ROI")
            return

        clamped = roi.clamp(source.shape[:2])
        if not clamped.is_valid():
            QMessageBox.warning(self, "提示", "ROI 非法，请重新选择。")
            return

        self.current_roi = clamped
        self.check_enable_roi.setChecked(True)
        self._update_roi_status()
        self._refresh_all_previews()
        self.statusBar().showMessage(f"ROI 已设置: x={clamped.x}, y={clamped.y}, w={clamped.w}, h={clamped.h}")

    def clear_roi(self) -> None:
        self.current_roi = None
        self.check_enable_roi.setChecked(False)
        self._update_roi_status()
        self._refresh_all_previews()
        self.statusBar().showMessage("ROI 已清除")

    def _update_roi_status(self) -> None:
        if self.check_enable_roi.isChecked() and self.current_roi is not None:
            r = self.current_roi
            self.lbl_roi_status.setText(f"启用 ({r.x}, {r.y}, {r.w}, {r.h})")
        elif self.check_enable_roi.isChecked():
            self.lbl_roi_status.setText("启用但未设置")
        else:
            self.lbl_roi_status.setText("未启用")

    def _update_metadata_status(self) -> None:
        if self.metadata_manager.has_unexported_changes:
            self.lbl_metadata_status.setText("未导出变更")
        elif self.metadata_manager.last_export_at:
            self.lbl_metadata_status.setText(f"已导出 ({self.metadata_manager.last_export_at})")
        else:
            self.lbl_metadata_status.setText("已导出")

    def _get_save_roi(self) -> Optional[ROI]:
        if not self.check_enable_roi.isChecked():
            return None

        if self.current_roi is None:
            raise RuntimeError("ROI 已启用但尚未设置，请先点击“设置 ROI”。")

        source = self.latest_color_bgr.get("middle")
        if source is None:
            source = self.latest_color_bgr.get("left")
        if source is None:
            source = self.latest_color_bgr.get("right")
        if source is None:
            raise RuntimeError("当前没有可用图像，无法应用 ROI。")

        clamped = self.current_roi.clamp(source.shape[:2])
        if not clamped.is_valid():
            raise RuntimeError("当前 ROI 非法，请重新设置。")
        return clamped

    def capture_and_save(self) -> None:
        for key in CAMERA_KEYS:
            if key not in self.latest_color_bgr or key not in self.latest_depth_u16:
                QMessageBox.warning(self, "提示", f"{CAMERA_TEXT[key]}路还没有可保存帧。")
                return
            if key not in self.camera_intrinsics_by_cam or key not in self.normals_estimators:
                QMessageBox.warning(self, "提示", f"{CAMERA_TEXT[key]}路内参不可用，请重启相机后重试。")
                return

        try:
            roi = self._get_save_roi()

            frames: Dict[str, Dict[str, np.ndarray]] = {}
            for key in CAMERA_KEYS:
                depth = self.latest_depth_u16[key]
                estimator = self.normals_estimators[key]
                intr = self.camera_intrinsics_by_cam[key]
                normals = estimator.compute_normals(depth, float(intr["depth_scale"]))
                normals_vis = NormalsEstimator.normals_to_vis(normals)

                frames[key] = {
                    "color": self.latest_color_bgr[key],
                    "depth": depth,
                    "normals": normals_vis,
                }

            record = self.saver.save_multi_sample(
                camera_frames=frames,
                light_label=self.get_selected_light_label(),
                roi=roi,
            )

            self.metadata_manager.add_record(record)
            self.btn_delete_last.setEnabled(True)
            self.lbl_last_save_status.setText(f"成功: {record['sample_name']}")
            self.lbl_sample_id_status.setText(f"{self.saver.next_sample_id():06d}")
            self._update_metadata_status()
            self.statusBar().showMessage(f"样本保存成功: {record['sample_name']}")

        except Exception as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            self.lbl_last_save_status.setText(f"失败: {exc}")

    def delete_last_sample(self) -> None:
        record = self.metadata_manager.get_last_record()
        if record is None:
            QMessageBox.information(self, "提示", "当前会话没有可删除样本。")
            return

        sample_name = record.get("sample_name", "unknown")
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确认删除最近样本 {sample_name} 及其左/中/右所有文件吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            self.saver.delete_sample_files(record)
            self.metadata_manager.pop_last_record()
            self.lbl_last_delete_status.setText(f"成功: {sample_name}")
            self.btn_delete_last.setEnabled(self.metadata_manager.record_count > 0)
            self._update_metadata_status()
            self.statusBar().showMessage(f"已删除样本: {sample_name}")
        except Exception as exc:
            QMessageBox.critical(self, "删除失败", str(exc))
            self.lbl_last_delete_status.setText(f"失败: {exc}")

    def export_metadata(self) -> bool:
        if self.metadata_manager.record_count == 0:
            QMessageBox.information(self, "提示", "当前会话暂无样本，无法导出 metadata。")
            return False

        if not self.metadata_manager.has_unexported_changes:
            QMessageBox.information(self, "提示", "当前没有新的 metadata 变更。")
            return False

        try:
            paths = self.metadata_manager.export(str(self.saver.root_dir))
            self._update_metadata_status()
            self.statusBar().showMessage(f"导出完成: {paths['metadata_json']} | {paths['index_csv']}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.workers:
            self.stop_camera()

        if self.metadata_manager.has_unexported_changes:
            reply = QMessageBox.question(
                self,
                "未导出 metadata",
                "检测到未导出的 metadata 变更，是否在退出前导出？",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes and not self.export_metadata():
                event.ignore()
                return

        event.accept()
