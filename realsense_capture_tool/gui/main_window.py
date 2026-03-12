from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRect, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
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
from utils.roi import ROI
from utils.saver import DatasetSaver

CAMERA_KEYS = ("left", "middle", "right")
CAMERA_TEXT = {
    "left": "左",
    "middle": "中",
    "right": "右",
}


class PreviewLabel(QLabel):
    roi_selected = pyqtSignal(str, object)  # camera_key, ROI

    def __init__(self, camera_key: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.camera_key = camera_key
        self.image_size: Optional[Tuple[int, int]] = None  # (w, h)
        self.active_roi: Optional[ROI] = None
        self.roi_edit_enabled: bool = False
        self._drag_start: Optional[QPoint] = None
        self._drag_current: Optional[QPoint] = None

    def set_image_size(self, image_w: int, image_h: int) -> None:
        self.image_size = (int(image_w), int(image_h))

    def set_active_roi(self, roi: Optional[ROI]) -> None:
        self.active_roi = roi
        self.update()

    def set_roi_edit_enabled(self, enabled: bool) -> None:
        self.roi_edit_enabled = bool(enabled)
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if not enabled:
            self._drag_start = None
            self._drag_current = None
        self.update()

    def _display_rect(self) -> Optional[QRect]:
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        x = (self.width() - pix.width()) // 2
        y = (self.height() - pix.height()) // 2
        return QRect(x, y, pix.width(), pix.height())

    def _label_rect_to_roi(self, rect: QRect) -> Optional[ROI]:
        if self.image_size is None:
            return None
        display = self._display_rect()
        if display is None or display.width() <= 0 or display.height() <= 0:
            return None

        clipped = rect.normalized().intersected(display)
        if clipped.isNull() or clipped.width() < 2 or clipped.height() < 2:
            return None

        image_w, image_h = self.image_size
        scale_x = image_w / float(display.width())
        scale_y = image_h / float(display.height())

        x = int(round((clipped.x() - display.x()) * scale_x))
        y = int(round((clipped.y() - display.y()) * scale_y))
        w = int(round(clipped.width() * scale_x))
        h = int(round(clipped.height() * scale_y))

        roi = ROI(x=x, y=y, w=w, h=h).clamp((image_h, image_w))
        return roi if roi.is_valid() else None

    def _roi_to_label_rect(self, roi: ROI) -> Optional[QRect]:
        if self.image_size is None:
            return None
        display = self._display_rect()
        if display is None or display.width() <= 0 or display.height() <= 0:
            return None

        image_w, image_h = self.image_size
        if image_w <= 0 or image_h <= 0:
            return None

        x = display.x() + int(round((roi.x / float(image_w)) * display.width()))
        y = display.y() + int(round((roi.y / float(image_h)) * display.height()))
        w = int(round((roi.w / float(image_w)) * display.width()))
        h = int(round((roi.h / float(image_h)) * display.height()))
        return QRect(x, y, max(1, w), max(1, h))

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self.roi_edit_enabled and event.button() == Qt.LeftButton:
            display = self._display_rect()
            if display is not None and display.contains(event.pos()):
                self._drag_start = event.pos()
                self._drag_current = event.pos()
                self.update()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.roi_edit_enabled and self._drag_start is not None:
            self._drag_current = event.pos()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self.roi_edit_enabled and self._drag_start is not None and event.button() == Qt.LeftButton:
            self._drag_current = event.pos()
            drag_rect = QRect(self._drag_start, self._drag_current).normalized()
            roi = self._label_rect_to_roi(drag_rect)
            self._drag_start = None
            self._drag_current = None
            self.update()
            if roi is not None:
                self.roi_selected.emit(self.camera_key, roi)
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.active_roi is not None:
            rect = self._roi_to_label_rect(self.active_roi)
            if rect is not None:
                painter.setPen(QPen(QColor(56, 201, 92), 2))
                painter.drawRect(rect)

        if self.roi_edit_enabled and self._drag_start is not None and self._drag_current is not None:
            drag_rect = QRect(self._drag_start, self._drag_current).normalized()
            painter.setPen(QPen(QColor(255, 196, 0), 2, Qt.DashLine))
            painter.drawRect(drag_rect)

        painter.end()


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
        self.pending_source_queue: list[str] = []
        self.pending_start_config: Optional[Tuple[int, int, int]] = None
        self.startup_failed: bool = False

        self.camera_intrinsics_by_cam: Dict[str, Dict] = {}
        self.normals_estimators: Dict[str, NormalsEstimator] = {}

        self.latest_color_bgr: Dict[str, np.ndarray] = {}
        self.latest_depth_u16: Dict[str, np.ndarray] = {}

        self.roi_by_camera: Dict[str, ROI] = {}
        self.roi_edit_mode: bool = False

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
            panel = self._make_camera_panel(key, f"{CAMERA_TEXT[key]}路相机")
            self.preview_views[key] = panel
            panel["image_label"].roi_selected.connect(self._on_preview_roi_selected)
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
        self.combo_resolution.addItems(["640x360", "640x480", "848x480", "1280x720"])
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
        self.btn_set_roi = QPushButton("设置 ROI（框选模式）")
        self.btn_set_roi.setCheckable(True)
        self.btn_clear_roi = QPushButton("清除全部 ROI")
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

    def _make_camera_panel(self, camera_key: str, title: str) -> Dict[str, QWidget]:
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)

        image_label = PreviewLabel(camera_key, "No Frame")
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
        self.btn_set_roi.toggled.connect(self._on_roi_edit_mode_toggled)
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
            usb_type = str(item.get("usb_type", "")).strip()
            if serial:
                filtered.append({"serial_number": serial, "name": name, "usb_type": usb_type})

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
                    usb_suffix = f" | USB:{dev['usb_type']}" if dev.get("usb_type") else ""
                    label = f"{dev['name']} | SN:{dev['serial_number']}{usb_suffix}"
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

    def _on_roi_toggled(self, enabled: bool) -> None:
        if not enabled and self.btn_set_roi.isChecked():
            self.btn_set_roi.setChecked(False)
        self._set_preview_roi_edit_enabled(enabled and self.roi_edit_mode)
        self._update_roi_status()
        self._refresh_all_previews()

    def _on_roi_edit_mode_toggled(self, enabled: bool) -> None:
        if enabled and not self.check_enable_roi.isChecked():
            self.check_enable_roi.setChecked(True)
        self.roi_edit_mode = bool(enabled)
        self._set_preview_roi_edit_enabled(self.check_enable_roi.isChecked() and self.roi_edit_mode)
        if self.roi_edit_mode:
            self.statusBar().showMessage("ROI 框选模式已开启：请在左/中/右预览框内拖拽设置各自 ROI。", 5000)
        else:
            self.statusBar().showMessage("ROI 框选模式已关闭。", 3000)

    def _set_preview_roi_edit_enabled(self, enabled: bool) -> None:
        for key in CAMERA_KEYS:
            label = self.preview_views[key]["image_label"]
            if isinstance(label, PreviewLabel):
                label.set_roi_edit_enabled(enabled)

    def _on_preview_roi_selected(self, camera_key: str, roi_obj: object) -> None:
        if not isinstance(roi_obj, ROI):
            return
        self.roi_by_camera[camera_key] = roi_obj
        self._update_roi_status()
        self._refresh_preview(camera_key)
        self.statusBar().showMessage(
            f"{CAMERA_TEXT.get(camera_key, camera_key)}路 ROI 已设置: "
            f"x={roi_obj.x}, y={roi_obj.y}, w={roi_obj.w}, h={roi_obj.h}",
            3500,
        )

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
        serial_to_device = {
            str(item.get("serial_number", "")).strip(): item
            for item in self.device_catalog
            if str(item.get("serial_number", "")).strip()
        }

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

        if len(unique_serials) > 1:
            multi_probe = RealSenseCamera.probe_multi_stream_start_safe(
                serial_numbers=unique_serials,
                width=width,
                height=height,
                fps=fps,
                timeout_sec=max(20.0, 8.0 * len(unique_serials)),
            )
            if not multi_probe.get("ok"):
                err_text = multi_probe.get("error", "未知错误")
                failed_serial = str(multi_probe.get("failed_serial", "")).strip()
                stage = str(multi_probe.get("stage", "")).strip() or "未知"
                per_camera_upper = RealSenseCamera.estimate_color_depth_stream_upper_bound_mbps(width, height, fps)
                total_upper = per_camera_upper * len(unique_serials)

                usb_lines = []
                for serial in unique_serials:
                    device = serial_to_device.get(serial, {})
                    usb_type = str(device.get("usb_type", "")).strip() or "未知"
                    usb_lines.append(f"SN:{serial} -> USB:{usb_type}")

                likely_causes = []
                if any("3" not in str(serial_to_device.get(serial, {}).get("usb_type", "")) for serial in unique_serials):
                    likely_causes.append("至少一台设备没有以 USB 3.x 枚举，Hub/线缆/转接器很可疑。")
                if total_upper >= 1000.0:
                    likely_causes.append(
                        f"当前模式理论上限约 {total_upper:.0f} Mbps，三机共享总线时已经接近高风险区。"
                    )
                if not likely_causes:
                    likely_causes.append("更像是 Hub 供电/USB 拓扑，或多设备并发初始化时序导致的问题。")

                failed_text = f"失败设备: SN:{failed_serial}\n" if failed_serial else ""
                usb_text = "\n".join(usb_lines)
                cause_text = "\n".join(likely_causes)
                QMessageBox.critical(
                    self,
                    "多机联合预检失败",
                    f"三路相机单台都能启动，但一起启动失败。\n"
                    f"这说明问题基本不在单台设备本身，而在共享 USB 链路/Hub/供电/并发启动时序。\n\n"
                    f"阶段: {stage}\n"
                    f"{failed_text}"
                    f"详细信息: {err_text}\n\n"
                    f"设备链路:\n{usb_text}\n\n"
                    f"判断:\n{cause_text}\n\n"
                    "建议先改用 640x360 @ 30 FPS，再确认三台都直连 USB 3.x 或接入独立供电 Hub。"
                )
                self.statusBar().showMessage(f"多机联合预检失败: {err_text}", 10000)
                return

        self.camera_intrinsics_by_cam.clear()
        self.normals_estimators.clear()
        self.latest_color_bgr.clear()
        self.latest_depth_u16.clear()
        self.started_sources.clear()
        self.pending_source_queue = list(unique_serials)
        self.pending_start_config = (width, height, fps)
        self.startup_failed = False

        self.btn_start_camera.setEnabled(False)
        self.btn_stop_camera.setEnabled(True)
        self.btn_capture_save.setEnabled(False)
        self.lbl_camera_status.setText(f"启动中: 0/{len(unique_serials)}（视图: 3）")
        self.statusBar().showMessage(
            f"相机启动中: 物理设备 {len(unique_serials)} 台，逻辑视图 3 路"
        )
        self._start_next_worker()

    def stop_camera(self) -> None:
        if not self.workers:
            self.pending_source_queue.clear()
            self.pending_start_config = None
            self.startup_failed = False
            return

        self.pending_source_queue.clear()
        self.pending_start_config = None

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
        self.startup_failed = False
        self.statusBar().showMessage("相机已停止")

    def _start_next_worker(self) -> None:
        if self.pending_start_config is None or not self.pending_source_queue:
            return

        width, height, fps = self.pending_start_config
        serial = self.pending_source_queue.pop(0)
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
        self.workers[serial] = worker
        worker.start()

    def _on_camera_started(self, source_key: str, intrinsics: Dict) -> None:
        self.started_sources[source_key] = intrinsics

        bound_views = self.source_to_views.get(source_key, [])
        for view_key in bound_views:
            self.camera_intrinsics_by_cam[view_key] = intrinsics
            self.normals_estimators[view_key] = NormalsEstimator.from_intrinsics_dict(intrinsics)

        started = len(self.started_sources)
        total = max(len(self.source_to_views), 1)
        self.lbl_camera_status.setText(f"运行中: {started}/{total}（视图: 3）")

        if self.pending_source_queue:
            self._start_next_worker()
            return

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
            self.pending_source_queue.clear()
            self.pending_start_config = None
            self.btn_start_camera.setEnabled(True)
            self.btn_stop_camera.setEnabled(False)
            self.btn_capture_save.setEnabled(False)
            self.lbl_camera_status.setText("已停止")

    def _on_worker_error(self, source_key: str, msg: str) -> None:
        if source_key not in self.started_sources and not self.startup_failed:
            self.startup_failed = True
            self.pending_source_queue.clear()
            bound_views = ",".join(CAMERA_TEXT.get(v, v) for v in self.source_to_views.get(source_key, []))
            extra = f"（{bound_views}）" if bound_views else ""
            QMessageBox.critical(
                self,
                "相机启动失败",
                f"设备 SN:{source_key}{extra} 在启动阶段失败。\n"
                f"详细信息: {msg}\n\n"
                "这更像是多机共享 USB 链路、Hub 供电，或并发初始化导致的问题。"
            )
            self.stop_camera()
            return

        bound_views = ",".join(CAMERA_TEXT.get(v, v) for v in self.source_to_views.get(source_key, []))
        if bound_views:
            self.statusBar().showMessage(f"设备 SN:{source_key}（{bound_views}）: {msg}", 6000)
        else:
            self.statusBar().showMessage(f"设备 SN:{source_key}: {msg}", 6000)
        for view_key in self.source_to_views.get(source_key, []):
            label = self.preview_views[view_key]["image_label"]
            if isinstance(label, QLabel):
                label.setText("取流失败，请查看状态栏错误信息")
                label.setPixmap(QPixmap())

    def _on_frame_ready(self, source_key: str, color_bgr: np.ndarray, depth_u16: np.ndarray) -> None:
        bound_views = self.source_to_views.get(source_key, [])
        if not bound_views:
            return

        for view_key in bound_views:
            self.latest_color_bgr[view_key] = color_bgr.copy()
            self.latest_depth_u16[view_key] = depth_u16.copy()
            self._refresh_preview(view_key)

    def _refresh_preview(self, camera_key: str) -> None:
        img = self.latest_color_bgr.get(camera_key)
        if img is None:
            return

        label = self.preview_views[camera_key]["image_label"]
        self._set_image_to_label(label, img)
        if isinstance(label, PreviewLabel):
            roi = self.roi_by_camera.get(camera_key) if self.check_enable_roi.isChecked() else None
            label.set_active_roi(roi)

    def _refresh_all_previews(self) -> None:
        for key in CAMERA_KEYS:
            self._refresh_preview(key)

    def _set_image_to_label(self, label: QLabel, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        bytes_per_line = c * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        target_size = label.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            target_size = label.minimumSize()
        pix = QPixmap.fromImage(qimg).scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setText("")
        label.setPixmap(pix)
        if isinstance(label, PreviewLabel):
            label.set_image_size(w, h)

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

    def clear_roi(self) -> None:
        self.roi_by_camera.clear()
        if self.btn_set_roi.isChecked():
            self.btn_set_roi.setChecked(False)
        self._update_roi_status()
        self._refresh_all_previews()
        self.statusBar().showMessage("三路 ROI 已清除", 3000)

    def _update_roi_status(self) -> None:
        if not self.check_enable_roi.isChecked():
            self.lbl_roi_status.setText("未启用")
            return

        parts = []
        for key in CAMERA_KEYS:
            roi = self.roi_by_camera.get(key)
            if roi is None:
                parts.append(f"{CAMERA_TEXT[key]}:未设置")
            else:
                parts.append(f"{CAMERA_TEXT[key]}:({roi.x},{roi.y},{roi.w},{roi.h})")
        suffix = " | 框选中" if self.roi_edit_mode else ""
        self.lbl_roi_status.setText("启用 " + " / ".join(parts) + suffix)

    def _update_metadata_status(self) -> None:
        if self.metadata_manager.has_unexported_changes:
            self.lbl_metadata_status.setText("未导出变更")
        elif self.metadata_manager.last_export_at:
            self.lbl_metadata_status.setText(f"已导出 ({self.metadata_manager.last_export_at})")
        else:
            self.lbl_metadata_status.setText("已导出")

    def _get_save_roi_by_camera(self) -> Dict[str, ROI]:
        if not self.check_enable_roi.isChecked():
            return {}

        missing = [CAMERA_TEXT[key] for key in CAMERA_KEYS if key not in self.roi_by_camera]
        if missing:
            raise RuntimeError(f"ROI 已启用，请先在预览框内设置这些相机的 ROI：{', '.join(missing)}。")

        result: Dict[str, ROI] = {}
        for key in CAMERA_KEYS:
            source = self.latest_color_bgr.get(key)
            if source is None:
                raise RuntimeError(f"{CAMERA_TEXT[key]}路当前没有可用图像，无法应用 ROI。")
            clamped = self.roi_by_camera[key].clamp(source.shape[:2])
            if not clamped.is_valid():
                raise RuntimeError(f"{CAMERA_TEXT[key]}路 ROI 非法，请重新设置。")
            result[key] = clamped
        return result

    def capture_and_save(self) -> None:
        for key in CAMERA_KEYS:
            if key not in self.latest_color_bgr or key not in self.latest_depth_u16:
                QMessageBox.warning(self, "提示", f"{CAMERA_TEXT[key]}路还没有可保存帧。")
                return
            if key not in self.camera_intrinsics_by_cam or key not in self.normals_estimators:
                QMessageBox.warning(self, "提示", f"{CAMERA_TEXT[key]}路内参不可用，请重启相机后重试。")
                return

        try:
            roi_by_camera = self._get_save_roi_by_camera()

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
                roi_by_camera=roi_by_camera,
            )

            self.metadata_manager.add_record(record)
            self.btn_delete_last.setEnabled(True)
            self.lbl_last_save_status.setText(f"成功: {record['sample_name']}")
            self.lbl_sample_id_status.setText(f"{self.saver.next_sample_id():06d}")
            self._update_metadata_status()
            self.statusBar().showMessage(f"样本保存成功: {record['sample_name']}", 3000)

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
            self.statusBar().showMessage(f"已删除样本: {sample_name}", 3000)
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
            self.statusBar().showMessage(f"导出完成: {paths['metadata_json']} | {paths['index_csv']}", 4000)
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
