from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
)

from calibration.chessboard_detector import ChessboardDetector, DetectionResult
from calibration.handeye_solver import (
    MODE_EYE_IN_HAND,
    MODE_EYE_TO_HAND,
    HandEyeResult,
    HandEyeSolver,
)
from calibration.transforms import matrix_to_xyzrpy, pose_xyzrpy_to_matrix
from camera.realsense_camera import RealSenseCamera
from data.sample_manager import SampleManager
from robot.ur_rtde_client import URRTDEClient, UR_RTDE_DEFAULT_PORT
from utils.io_utils import ensure_dir, read_json, write_json, write_matrix_txt, write_yaml


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("机器人手眼标定工具 (RealSense D435 + Chessboard)")
        self.resize(1600, 940)

        self.camera = RealSenseCamera()
        self.detector = ChessboardDetector(columns=9, rows=6, square_size=0.025)
        self.sample_manager = SampleManager()
        self.solver = HandEyeSolver()
        self.ur_robot = URRTDEClient()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.update_frame)
        self.ur_timer = QTimer(self)
        self.ur_timer.setInterval(100)
        self.ur_timer.timeout.connect(self.poll_ur_pose)

        self.current_frame: Optional[np.ndarray] = None
        self.last_detected_frame: Optional[np.ndarray] = None
        self.current_detection: Optional[DetectionResult] = None
        self.last_display_image: Optional[np.ndarray] = None

        self.current_intrinsics: Dict[str, Any] = {}
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.current_robot_pose_matrix: Optional[np.ndarray] = None
        self.last_ur_pose_vector: Optional[np.ndarray] = None
        self.last_ur_timestamp_s: Optional[float] = None

        self.last_result: Optional[HandEyeResult | Dict[str, Any]] = None
        self.last_frame_error_msg: str = ""

        self._build_ui()
        self._set_default_values()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)

        preview_group = QGroupBox("相机预览")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_label = QLabel("相机画面")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_label.setStyleSheet("background-color: #111; color: #ddd; border: 1px solid #333;")
        preview_layout.addWidget(self.camera_label)
        left_splitter.addWidget(preview_group)

        button_group = QGroupBox("控制")
        button_layout = QGridLayout(button_group)
        button_layout.setHorizontalSpacing(8)
        button_layout.setVerticalSpacing(8)

        self.btn_open_camera = QPushButton("打开相机")
        self.btn_close_camera = QPushButton("关闭相机")
        self.btn_detect = QPushButton("检测棋盘格")
        self.btn_capture = QPushButton("采集样本")
        self.btn_delete_sample = QPushButton("删除样本")
        self.btn_clear_samples = QPushButton("清空样本")
        self.btn_calibrate = QPushButton("开始标定")
        self.btn_save_project = QPushButton("保存工程")
        self.btn_load_project = QPushButton("加载工程")
        self.btn_export_result = QPushButton("导出结果")

        buttons = [
            self.btn_open_camera,
            self.btn_close_camera,
            self.btn_detect,
            self.btn_capture,
            self.btn_delete_sample,
            self.btn_clear_samples,
            self.btn_calibrate,
            self.btn_save_project,
            self.btn_load_project,
            self.btn_export_result,
        ]
        for idx, button in enumerate(buttons):
            row = idx // 5
            col = idx % 5
            button.setMinimumHeight(40)
            button_layout.addWidget(button, row, col)
        for col in range(5):
            button_layout.setColumnStretch(col, 1)
        left_splitter.addWidget(button_group)
        left_splitter.setStretchFactor(0, 5)
        left_splitter.setStretchFactor(1, 1)
        left_splitter.setSizes([700, 220])

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout(param_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Eye-in-Hand（相机安装在末端）", MODE_EYE_IN_HAND)
        self.mode_combo.addItem("Eye-to-Hand（相机固定在环境）", MODE_EYE_TO_HAND)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Tsai", "Park", "Daniilidis"])

        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(2, 30)
        self.columns_spin.setValue(9)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 30)
        self.rows_spin.setValue(6)

        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(0.001, 1.0)
        self.square_size_spin.setDecimals(4)
        self.square_size_spin.setSingleStep(0.001)
        self.square_size_spin.setValue(0.025)
        self.square_size_spin.setSuffix(" m")

        param_layout.addRow("标定模式", self.mode_combo)
        param_layout.addRow("求解方法", self.method_combo)
        param_layout.addRow("棋盘格列数(内角点)", self.columns_spin)
        param_layout.addRow("棋盘格行数(内角点)", self.rows_spin)
        param_layout.addRow("方块尺寸", self.square_size_spin)

        right_layout.addWidget(param_group)

        ur_group = QGroupBox("UR 机器人接口")
        ur_layout = QGridLayout(ur_group)

        self.ur_host_edit = QLineEdit()
        self.ur_host_edit.setPlaceholderText("例如 192.168.0.10")

        self.ur_port_spin = QSpinBox()
        self.ur_port_spin.setRange(1, 65535)
        self.ur_port_spin.setValue(UR_RTDE_DEFAULT_PORT)

        self.ur_frequency_spin = QDoubleSpinBox()
        self.ur_frequency_spin.setRange(1.0, 500.0)
        self.ur_frequency_spin.setDecimals(1)
        self.ur_frequency_spin.setSingleStep(1.0)
        self.ur_frequency_spin.setValue(10.0)
        self.ur_frequency_spin.setSuffix(" Hz")

        self.btn_scan_ur = QPushButton("检索端口")
        self.btn_connect_ur = QPushButton("连接UR")
        self.btn_disconnect_ur = QPushButton("断开UR")
        self.btn_read_ur_pose = QPushButton("读取TCP位姿")
        self.ur_status_label = QLabel("状态: 未连接")
        self.ur_status_label.setWordWrap(True)
        self.ur_ports_label = QLabel("开放端口: 未检索")
        self.ur_ports_label.setWordWrap(True)

        ur_layout.addWidget(QLabel("UR IP"), 0, 0)
        ur_layout.addWidget(self.ur_host_edit, 0, 1, 1, 3)
        ur_layout.addWidget(QLabel("RTDE端口"), 1, 0)
        ur_layout.addWidget(self.ur_port_spin, 1, 1)
        ur_layout.addWidget(QLabel("频率"), 1, 2)
        ur_layout.addWidget(self.ur_frequency_spin, 1, 3)
        ur_layout.addWidget(self.btn_scan_ur, 2, 0)
        ur_layout.addWidget(self.btn_connect_ur, 2, 1)
        ur_layout.addWidget(self.btn_disconnect_ur, 2, 2)
        ur_layout.addWidget(self.btn_read_ur_pose, 2, 3)
        ur_layout.addWidget(self.ur_status_label, 3, 0, 1, 4)
        ur_layout.addWidget(self.ur_ports_label, 4, 0, 1, 4)

        right_layout.addWidget(ur_group)

        pose_group = QGroupBox("机器人位姿输入 / 显示 (T_base_tool)")
        pose_layout = QGridLayout(pose_group)

        self.pose_x = self._make_spin(-10.0, 10.0, 0.001, 4, " m")
        self.pose_y = self._make_spin(-10.0, 10.0, 0.001, 4, " m")
        self.pose_z = self._make_spin(-10.0, 10.0, 0.001, 4, " m")
        self.pose_roll = self._make_spin(-360.0, 360.0, 1.0, 3, " deg")
        self.pose_pitch = self._make_spin(-360.0, 360.0, 1.0, 3, " deg")
        self.pose_yaw = self._make_spin(-360.0, 360.0, 1.0, 3, " deg")

        pose_layout.addWidget(QLabel("x"), 0, 0)
        pose_layout.addWidget(self.pose_x, 0, 1)
        pose_layout.addWidget(QLabel("y"), 0, 2)
        pose_layout.addWidget(self.pose_y, 0, 3)
        pose_layout.addWidget(QLabel("z"), 0, 4)
        pose_layout.addWidget(self.pose_z, 0, 5)
        pose_layout.addWidget(QLabel("roll"), 1, 0)
        pose_layout.addWidget(self.pose_roll, 1, 1)
        pose_layout.addWidget(QLabel("pitch"), 1, 2)
        pose_layout.addWidget(self.pose_pitch, 1, 3)
        pose_layout.addWidget(QLabel("yaw"), 1, 4)
        pose_layout.addWidget(self.pose_yaw, 1, 5)

        self.pose_source_label = QLabel("位姿来源: 手动输入 (xyz + rpy)")
        self.pose_source_label.setWordWrap(True)
        self.ur_raw_pose_label = QLabel("UR 原始TCP: -")
        self.ur_raw_pose_label.setWordWrap(True)
        pose_layout.addWidget(self.pose_source_label, 2, 0, 1, 6)
        pose_layout.addWidget(self.ur_raw_pose_label, 3, 0, 1, 6)

        right_layout.addWidget(pose_group)

        sample_group = QGroupBox("样本列表")
        sample_layout = QVBoxLayout(sample_group)
        self.sample_table = QTableWidget(0, 9)
        self.sample_table.setHorizontalHeaderLabels(
            ["ID", "时间", "x", "y", "z", "roll", "pitch", "yaw", "reproj(px)"]
        )
        self.sample_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sample_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sample_table.verticalHeader().setVisible(False)
        self.sample_table.setAlternatingRowColors(True)
        self.sample_table.horizontalHeader().setStretchLastSection(True)
        sample_layout.addWidget(self.sample_table)
        right_layout.addWidget(sample_group, stretch=1)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setWidget(right_panel)

        splitter.addWidget(left_splitter)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1100, 600])

        self.btn_open_camera.clicked.connect(self.open_camera)
        self.btn_close_camera.clicked.connect(self.close_camera)
        self.btn_detect.clicked.connect(self.detect_chessboard)
        self.btn_capture.clicked.connect(self.capture_sample)
        self.btn_delete_sample.clicked.connect(self.delete_sample)
        self.btn_clear_samples.clicked.connect(self.clear_samples)
        self.btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_save_project.clicked.connect(self.save_project)
        self.btn_load_project.clicked.connect(self.load_project)
        self.btn_export_result.clicked.connect(self.export_result)
        self.btn_scan_ur.clicked.connect(self.scan_ur_ports)
        self.btn_connect_ur.clicked.connect(self.connect_ur)
        self.btn_disconnect_ur.clicked.connect(lambda: self.disconnect_ur())
        self.btn_read_ur_pose.clicked.connect(lambda: self.read_ur_pose())

    def _set_default_values(self) -> None:
        self.mode_combo.setCurrentIndex(0)
        self.method_combo.setCurrentText("Tsai")
        self.ur_host_edit.setText("192.168.0.10")
        for spin in [
            self.pose_x,
            self.pose_y,
            self.pose_z,
            self.pose_roll,
            self.pose_pitch,
            self.pose_yaw,
        ]:
            spin.valueChanged.connect(self._on_manual_pose_changed)
        self.log("程序已启动。默认棋盘格参数: 9 x 6, 方块 0.025 m")
        self.log("UR 接口基于官方 RTDE Python Client Library，读取字段: actual_TCP_pose (x,y,z,rx,ry,rz)")
        if not self.ur_robot.is_sdk_available():
            self._update_ur_status("官方 RTDE SDK 未安装")
            self.log(f"UR 官方 RTDE SDK 未安装: {self.ur_robot.sdk_error_message()}")

    def _make_spin(self, min_val: float, max_val: float, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _pose_spins(self) -> list[QDoubleSpinBox]:
        return [
            self.pose_x,
            self.pose_y,
            self.pose_z,
            self.pose_roll,
            self.pose_pitch,
            self.pose_yaw,
        ]

    def _set_pose_inputs_enabled(self, enabled: bool) -> None:
        for spin in self._pose_spins():
            spin.setEnabled(enabled)

    def _on_manual_pose_changed(self, _value: float) -> None:
        if self.ur_robot.is_connected:
            return
        self.current_robot_pose_matrix = None
        self.pose_source_label.setText("位姿来源: 手动输入 (xyz + rpy)")

    def _update_ur_status(self, message: str) -> None:
        self.ur_status_label.setText(f"状态: {message}")

    def _apply_robot_pose(
        self,
        transform: np.ndarray,
        source_text: str,
        ur_pose_vector: Optional[np.ndarray] = None,
        timestamp_s: Optional[float] = None,
    ) -> None:
        self.current_robot_pose_matrix = np.asarray(transform, dtype=np.float64)
        x, y, z, roll, pitch, yaw = matrix_to_xyzrpy(self.current_robot_pose_matrix, degrees=True)
        values = [x, y, z, roll, pitch, yaw]
        for spin, value in zip(self._pose_spins(), values):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

        source = source_text
        if timestamp_s is not None:
            source = f"{source_text} | RTDE timestamp={timestamp_s:.4f}s"
        self.pose_source_label.setText(f"位姿来源: {source}")

        if ur_pose_vector is None:
            self.ur_raw_pose_label.setText("UR 原始TCP: -")
            self.last_ur_pose_vector = None
            self.last_ur_timestamp_s = None
            return

        self.last_ur_pose_vector = np.asarray(ur_pose_vector, dtype=np.float64).reshape(6)
        self.last_ur_timestamp_s = timestamp_s
        pose_text = ", ".join(f"{value:.6f}" for value in self.last_ur_pose_vector.tolist())
        self.ur_raw_pose_label.setText(f"UR 原始TCP (m,rad): [{pose_text}]")

    def _to_qimage(self, image_bgr: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        bytes_per_line = c * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def _display_image(self, image_bgr: np.ndarray) -> None:
        self.last_display_image = image_bgr.copy()
        qimage = self._to_qimage(image_bgr)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.last_display_image is not None:
            self._display_image(self.last_display_image)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.close_camera()
        self.disconnect_ur(log_message=False)
        super().closeEvent(event)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _update_detector_pattern(self) -> None:
        self.detector.set_pattern(
            columns=self.columns_spin.value(),
            rows=self.rows_spin.value(),
            square_size=self.square_size_spin.value(),
        )

    def _update_intrinsics(self, intrinsics: Dict[str, Any]) -> None:
        self.current_intrinsics = dict(intrinsics) if intrinsics else {}
        required = {"fx", "fy", "ppx", "ppy"}
        if required.issubset(self.current_intrinsics.keys()):
            self.camera_matrix = np.array(
                [
                    [self.current_intrinsics["fx"], 0.0, self.current_intrinsics["ppx"]],
                    [0.0, self.current_intrinsics["fy"], self.current_intrinsics["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            coeffs = np.array(self.current_intrinsics.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
            if coeffs.size < 5:
                coeffs = np.pad(coeffs, (0, 5 - coeffs.size), mode="constant")
            self.dist_coeffs = coeffs.reshape(-1, 1)
        else:
            self.camera_matrix = None
            self.dist_coeffs = None

    def update_frame(self) -> None:
        frame, err = self.camera.get_frame()
        if frame is None:
            if err and err != self.last_frame_error_msg:
                self.last_frame_error_msg = err
                self.log(err)
            if not self.camera.is_open:
                self.timer.stop()
                self.log("相机已断开，预览已停止。")
            return

        self.last_frame_error_msg = ""
        self.current_frame = frame
        self._display_image(frame)

    def open_camera(self) -> None:
        self._update_detector_pattern()

        ok, msg = self.camera.open()
        self.log(msg)
        if not ok:
            QMessageBox.warning(self, "相机错误", msg)
            return

        self._update_intrinsics(self.camera.get_intrinsics_dict())
        if self.camera_matrix is None:
            self.log("警告: 未读取到相机内参，solvePnP 将不可用。")
        else:
            intr = self.current_intrinsics
            self.log(
                "相机内参: "
                f"fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, ppx={intr['ppx']:.2f}, ppy={intr['ppy']:.2f}"
            )

        self.timer.start()

    def close_camera(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.camera.is_open:
            self.camera.close()
            self.log("相机已关闭。")

    def _get_ur_host(self) -> str:
        return self.ur_host_edit.text().strip()

    def _update_ur_timer_interval(self) -> None:
        requested_hz = max(1.0, self.ur_frequency_spin.value())
        poll_hz = min(requested_hz, 50.0)
        self.ur_timer.setInterval(max(20, int(round(1000.0 / poll_hz))))

    def scan_ur_ports(self) -> None:
        host = self._get_ur_host()
        if not host:
            self.log("请输入 UR 机器人的 IP 地址。")
            return

        try:
            port_statuses = self.ur_robot.scan_known_ports(host)
        except Exception as exc:
            message = f"UR 端口检索失败: {exc}"
            self.log(message)
            QMessageBox.warning(self, "UR 检索失败", message)
            return

        open_ports = [item.port for item in port_statuses if item.is_open]
        self.ur_ports_label.setText(f"开放端口: {self.ur_robot.describe_open_ports(port_statuses)}")
        if UR_RTDE_DEFAULT_PORT in open_ports:
            self.ur_port_spin.setValue(UR_RTDE_DEFAULT_PORT)
            self.log(f"UR 端口检索完成: 已发现 RTDE 端口 {UR_RTDE_DEFAULT_PORT}。")
        else:
            self.log(
                "UR 端口检索完成: 未发现 RTDE 端口 30004。"
                " 若机器人在线，请检查 PolyScope 安全设置中的 RTDE 服务和入站访问策略。"
            )

    def connect_ur(self) -> None:
        host = self._get_ur_host()
        if not host:
            self.log("请输入 UR 机器人的 IP 地址。")
            return

        self._update_ur_status("连接中")
        try:
            controller_version = self.ur_robot.connect(
                host=host,
                port=self.ur_port_spin.value(),
                frequency_hz=self.ur_frequency_spin.value(),
            )
        except Exception as exc:
            message = f"连接 UR 失败: {exc}"
            self._update_ur_status("连接失败")
            self.log(message)
            QMessageBox.warning(self, "UR 连接失败", message)
            return

        version_text = ".".join(str(item) for item in controller_version)
        self._set_pose_inputs_enabled(False)
        self._update_ur_timer_interval()
        self.ur_timer.start()
        self._update_ur_status(f"已连接 {host}:{self.ur_robot.port} | PolyScope {version_text}")
        self.log(
            f"已连接 UR RTDE: {host}:{self.ur_robot.port}, 频率={self.ur_robot.frequency_hz:.1f} Hz, "
            f"控制器版本={version_text}"
        )
        self.log("说明: actual_TCP_pose 是 UR 的 TCP 位姿 (x,y,z,rx,ry,rz)，其中姿态为旋转向量。")
        self.read_ur_pose(log_success=True, blocking=True)

    def disconnect_ur(self, log_message: bool = True) -> None:
        was_connected = self.ur_robot.is_connected
        if self.ur_timer.isActive():
            self.ur_timer.stop()
        self.ur_robot.disconnect()
        self.current_robot_pose_matrix = None
        self.last_ur_pose_vector = None
        self.last_ur_timestamp_s = None
        self.ur_raw_pose_label.setText("UR 原始TCP: -")
        self.pose_source_label.setText("位姿来源: 手动输入 (当前显示值已保留，可继续编辑)")
        self._set_pose_inputs_enabled(True)
        self._update_ur_status("未连接")
        if log_message and was_connected:
            self.log("UR RTDE 已断开。")

    def poll_ur_pose(self) -> None:
        if not self.ur_robot.is_connected:
            return
        try:
            sample = self.ur_robot.read_tcp_pose(blocking=False)
        except Exception as exc:
            self.log(f"UR 轮询失败，连接已断开: {exc}")
            self.disconnect_ur(log_message=False)
            return

        if sample is None:
            return

        self._apply_robot_pose(
            sample.transform,
            source_text="UR RTDE 实时位姿",
            ur_pose_vector=sample.pose_vector,
            timestamp_s=sample.timestamp_s,
        )

    def read_ur_pose(self, log_success: bool = True, blocking: bool = True) -> bool:
        if not self.ur_robot.is_connected:
            self.log("UR 尚未连接。")
            return False

        try:
            sample = self.ur_robot.read_tcp_pose(blocking=blocking)
        except Exception as exc:
            self.log(f"读取 UR TCP 位姿失败: {exc}")
            self.disconnect_ur(log_message=False)
            return False

        if sample is None:
            if log_success:
                self.log("当前没有新的 UR RTDE 数据包。")
            return False

        self._apply_robot_pose(
            sample.transform,
            source_text="UR TCP 位姿",
            ur_pose_vector=sample.pose_vector,
            timestamp_s=sample.timestamp_s,
        )

        if log_success:
            x, y, z, roll, pitch, yaw = matrix_to_xyzrpy(sample.transform, degrees=True)
            self.log(
                "已读取 UR TCP 位姿: "
                f"x={x:.4f}, y={y:.4f}, z={z:.4f} m, "
                f"roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f} deg"
            )
        return True

    def detect_chessboard(self) -> None:
        if self.current_frame is None:
            self.log("当前没有可用图像，请先打开相机。")
            return

        if self.camera_matrix is None or self.dist_coeffs is None:
            self.log("缺少相机内参，无法进行 solvePnP。")
            return

        self._update_detector_pattern()
        frame_for_detection = self.current_frame.copy()
        result = self.detector.detect(frame_for_detection, self.camera_matrix, self.dist_coeffs)
        self.current_detection = result

        self._display_image(result.visualization)
        if result.success:
            self.last_detected_frame = frame_for_detection
            t = result.tvec.reshape(3) if result.tvec is not None else np.zeros(3)
            self.log(
                f"检测成功: reproj={result.reproj_error:.4f}px, "
                f"t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m"
            )
        else:
            self.last_detected_frame = None
            self.log(f"检测失败: {result.message}")

    def _get_robot_pose_matrix(self) -> np.ndarray:
        if self.current_robot_pose_matrix is not None:
            return np.asarray(self.current_robot_pose_matrix, dtype=np.float64).copy()
        return pose_xyzrpy_to_matrix(
            x=self.pose_x.value(),
            y=self.pose_y.value(),
            z=self.pose_z.value(),
            roll=self.pose_roll.value(),
            pitch=self.pose_pitch.value(),
            yaw=self.pose_yaw.value(),
            degrees=True,
        )

    def capture_sample(self) -> None:
        if self.current_detection is None or not self.current_detection.success:
            self.log("请先成功检测棋盘格后再采集样本。")
            return

        if self.last_detected_frame is None:
            self.log("缺少检测时图像，无法采集。")
            return

        if (
            self.current_detection.target_pose_cam is None
            or self.current_detection.rvec is None
            or self.current_detection.tvec is None
        ):
            self.log("检测结果不完整，无法采集。")
            return

        if self.ur_robot.is_connected and not self.read_ur_pose(log_success=False, blocking=True):
            self.log("采样前刷新 UR TCP 位姿失败，本次采集已取消。")
            return

        robot_pose = self._get_robot_pose_matrix()
        sample = self.sample_manager.add_sample(
            image=self.last_detected_frame,
            robot_pose=robot_pose,
            target_pose_cam=self.current_detection.target_pose_cam,
            rvec=self.current_detection.rvec,
            tvec=self.current_detection.tvec,
            reproj_error=self.current_detection.reproj_error,
        )
        self._refresh_sample_table()
        pose_source = "UR TCP" if self.ur_robot.is_connected else "手动输入"
        self.log(
            f"已采集样本 #{sample.sample_id}，位姿来源={pose_source}，当前样本数: {len(self.sample_manager.samples)}"
        )

    def delete_sample(self) -> None:
        rows = self.sample_table.selectionModel().selectedRows()
        if not rows:
            self.log("请先在样本列表中选择要删除的样本。")
            return
        row = rows[0].row()
        sample_id = self.sample_manager.samples[row].sample_id
        self.sample_manager.remove_sample(row)
        self._refresh_sample_table()
        self.log(f"已删除样本 #{sample_id}")

    def clear_samples(self) -> None:
        if not self.sample_manager.samples:
            self.log("样本列表为空。")
            return

        answer = QMessageBox.question(self, "确认", "确定清空全部样本吗？")
        if answer != QMessageBox.Yes:
            return

        self.sample_manager.clear()
        self._refresh_sample_table()
        self.log("已清空所有样本。")

    def _refresh_sample_table(self) -> None:
        samples = self.sample_manager.samples
        self.sample_table.setRowCount(len(samples))

        for row, sample in enumerate(samples):
            pose = np.asarray(sample.robot_pose, dtype=np.float64)
            x, y, z, roll, pitch, yaw = matrix_to_xyzrpy(pose, degrees=True)
            values = [
                str(sample.sample_id),
                sample.timestamp,
                f"{x:.4f}",
                f"{y:.4f}",
                f"{z:.4f}",
                f"{roll:.2f}",
                f"{pitch:.2f}",
                f"{yaw:.2f}",
                f"{sample.reproj_error:.4f}",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.sample_table.setItem(row, col, item)

        self.sample_table.resizeColumnsToContents()

    def start_calibration(self) -> None:
        samples = self.sample_manager.get_samples()
        if len(samples) < 3:
            self.log("样本不足，至少需要 3 个样本。")
            return

        mode = str(self.mode_combo.currentData())
        method = self.method_combo.currentText()

        try:
            result = self.solver.solve(samples, mode=mode, method=method)
        except Exception as exc:
            self.log(f"标定失败: {exc}")
            QMessageBox.warning(self, "标定失败", str(exc))
            return

        self.last_result = result
        self._log_calibration_result(result)

    def _log_calibration_result(self, result: HandEyeResult) -> None:
        self.log(f"标定成功: mode={result.mode}, method={result.method}")
        self.log(f"结果变换: {result.transform_name}")
        self.log(np.array2string(result.transform_matrix, precision=6, suppress_small=True))
        self.log(
            "平移(m): "
            f"[{result.translation[0]:.6f}, {result.translation[1]:.6f}, {result.translation[2]:.6f}]"
        )
        self.log(
            "欧拉角(deg xyz): "
            f"[{result.euler_deg[0]:.4f}, {result.euler_deg[1]:.4f}, {result.euler_deg[2]:.4f}]"
        )
        self.log(
            "四元数(xyzw): "
            f"[{result.quaternion_xyzw[0]:.6f}, {result.quaternion_xyzw[1]:.6f}, "
            f"{result.quaternion_xyzw[2]:.6f}, {result.quaternion_xyzw[3]:.6f}]"
        )

        self.log(
            "误差统计: "
            f"平均平移误差={result.mean_translation_error_m:.6f} m, "
            f"最大平移误差={result.max_translation_error_m:.6f} m, "
            f"平均旋转误差={result.mean_rotation_error_deg:.4f} deg, "
            f"最大旋转误差={result.max_rotation_error_deg:.4f} deg"
        )

        for item in result.per_sample_errors:
            self.log(
                f"样本#{item['sample_id']}: "
                f"trans_err={item['translation_error_m']:.6f} m, "
                f"rot_err={item['rotation_error_deg']:.4f} deg, "
                f"reproj={item['reprojection_error_px']:.4f}px"
            )

    def _collect_config(self) -> Dict[str, Any]:
        return {
            "mode": str(self.mode_combo.currentData()),
            "method": self.method_combo.currentText(),
            "chessboard": {
                "columns": self.columns_spin.value(),
                "rows": self.rows_spin.value(),
                "square_size": self.square_size_spin.value(),
            },
            "ur": {
                "host": self._get_ur_host(),
                "rtde_port": self.ur_port_spin.value(),
                "frequency_hz": self.ur_frequency_spin.value(),
            },
        }

    def _apply_config(self, config: Dict[str, Any]) -> None:
        mode = config.get("mode", MODE_EYE_IN_HAND)
        method = config.get("method", "Tsai")
        chessboard = config.get("chessboard", {})

        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

        if method in ["Tsai", "Park", "Daniilidis"]:
            self.method_combo.setCurrentText(method)

        self.columns_spin.setValue(int(chessboard.get("columns", 9)))
        self.rows_spin.setValue(int(chessboard.get("rows", 6)))
        self.square_size_spin.setValue(float(chessboard.get("square_size", 0.025)))
        ur_config = config.get("ur", {})
        self.ur_host_edit.setText(str(ur_config.get("host", self.ur_host_edit.text())))
        self.ur_port_spin.setValue(int(ur_config.get("rtde_port", UR_RTDE_DEFAULT_PORT)))
        self.ur_frequency_spin.setValue(float(ur_config.get("frequency_hz", 10.0)))

        self._update_detector_pattern()

    def _result_to_dict(self) -> Optional[Dict[str, Any]]:
        if self.last_result is None:
            return None
        if isinstance(self.last_result, HandEyeResult):
            return self.last_result.to_dict()
        if isinstance(self.last_result, dict):
            return self.last_result
        return None

    def save_project(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(self, "选择工程保存目录")
        if not project_dir:
            return

        project_path = ensure_dir(project_dir)
        self.sample_manager.save_samples(project_path)
        write_json(project_path / "config.json", self._collect_config())

        intrinsics = self.camera.get_intrinsics_dict() if self.camera.is_open else self.current_intrinsics
        write_json(project_path / "intrinsics.json", intrinsics)

        result_data = self._result_to_dict()
        write_json(project_path / "result.json", result_data if result_data is not None else {})

        self.log(f"工程已保存: {project_path}")

    def load_project(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(self, "选择工程目录")
        if not project_dir:
            return

        project_path = Path(project_dir)
        config = read_json(project_path / "config.json", default={})
        intrinsics = read_json(project_path / "intrinsics.json", default={})
        result_data = read_json(project_path / "result.json", default=None)

        self._apply_config(config)
        self._update_intrinsics(intrinsics)
        self.sample_manager.load_samples(project_path)
        self._refresh_sample_table()
        self.last_result = result_data

        self.log(f"工程已加载: {project_path}")
        self.log(f"加载样本数量: {len(self.sample_manager.samples)}")

        if intrinsics:
            self.log("已加载相机内参。")
        if result_data is not None:
            self.log("已加载历史标定结果。")

    def export_result(self) -> None:
        result_data = self._result_to_dict()
        if result_data is None:
            self.log("当前没有可导出的标定结果。")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出结果",
            "handeye_result.json",
            "JSON (*.json);;YAML (*.yaml *.yml);;TXT Matrix (*.txt)",
        )
        if not path:
            return

        output_path = Path(path)

        if selected_filter.startswith("JSON"):
            if output_path.suffix.lower() != ".json":
                output_path = output_path.with_suffix(".json")
            write_json(output_path, result_data)
        elif selected_filter.startswith("YAML"):
            if output_path.suffix.lower() not in {".yaml", ".yml"}:
                output_path = output_path.with_suffix(".yaml")
            write_yaml(output_path, result_data)
        else:
            if output_path.suffix.lower() != ".txt":
                output_path = output_path.with_suffix(".txt")
            matrix = np.asarray(result_data.get("transform_matrix", np.eye(4)), dtype=np.float64)
            name = str(result_data.get("transform_name", "T"))
            write_matrix_txt(output_path, matrix, name=name)

        self.log(f"结果已导出: {output_path}")
