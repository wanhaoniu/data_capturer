from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
    _RS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local runtime
    rs = None  # type: ignore[assignment]
    _RS_IMPORT_ERROR = exc


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    serial_number: str = ""


class RealSenseCamera:
    """Intel RealSense D435 wrapper with explicit depth-to-color alignment."""

    def __init__(self) -> None:
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None
        self.align: Optional[rs.align] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.depth_scale: Optional[float] = None
        self.serial_number: str = ""
        self.intrinsics: Optional[Dict] = None
        self._running: bool = False

    @staticmethod
    def _ensure_rs_available() -> None:
        if rs is None:
            detail = f": {_RS_IMPORT_ERROR}" if _RS_IMPORT_ERROR else ""
            raise RuntimeError(f"pyrealsense2 不可用{detail}")

    @staticmethod
    def has_device() -> bool:
        RealSenseCamera._ensure_rs_available()
        context = rs.context()
        return len(context.query_devices()) > 0

    @staticmethod
    def first_device_info() -> Dict[str, str]:
        RealSenseCamera._ensure_rs_available()
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            return {}
        dev = devices[0]
        info = {}
        for key, name in [
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
        ]:
            try:
                info[name] = dev.get_info(key)
            except Exception:
                info[name] = ""
        return info

    @staticmethod
    def probe_device_info_safe(timeout_sec: float = 3.0) -> Dict:
        """
        Probe RealSense backend in a subprocess to avoid hard crash in GUI process.
        """
        probe_code = """
import json
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
info = {}
if len(devices) > 0:
    try:
        dev = devices[0]
        for key, name in [
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
        ]:
            try:
                info[name] = dev.get_info(key)
            except Exception:
                info[name] = ""
    except Exception:
        # On some macOS/runtime combinations, touching device info can fail
        # while query_devices still reports connected hardware.
        info = {}
print(json.dumps({"device_count": int(len(devices)), "info": info}, ensure_ascii=False))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "backend_ok": False,
                "device_found": False,
                "device_count": 0,
                "info": {},
                "error": "RealSense 探测超时。",
            }

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            err = stderr if stderr else f"子进程异常退出 (code={completed.returncode})"
            return {
                "backend_ok": False,
                "device_found": False,
                "device_count": 0,
                "info": {},
                "error": err,
            }

        stdout = completed.stdout.strip().splitlines()
        if not stdout:
            return {
                "backend_ok": False,
                "device_found": False,
                "device_count": 0,
                "info": {},
                "error": "RealSense 探测未返回结果。",
            }

        try:
            payload = json.loads(stdout[-1])
            count = int(payload.get("device_count", 0))
            info = payload.get("info", {}) if isinstance(payload.get("info"), dict) else {}
            return {
                "backend_ok": True,
                "device_found": count > 0,
                "device_count": count,
                "info": info,
                "error": "",
            }
        except Exception as exc:
            return {
                "backend_ok": False,
                "device_found": False,
                "device_count": 0,
                "info": {},
                "error": f"解析探测结果失败: {exc}",
            }

    @staticmethod
    def probe_device_list_safe(timeout_sec: float = 3.0) -> Dict:
        """
        Return connected RealSense device list in subprocess.
        """
        probe_code = """
import json
import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()
items = []
for i in range(int(len(devices))):
    entry = {"name": "", "serial_number": "", "firmware_version": ""}
    try:
        dev = devices[i]
        for key, field in [
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
        ]:
            try:
                entry[field] = dev.get_info(key)
            except Exception:
                entry[field] = ""
    except Exception:
        pass
    items.append(entry)

print(json.dumps({"device_count": int(len(devices)), "devices": items}, ensure_ascii=False))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "backend_ok": False,
                "device_count": 0,
                "devices": [],
                "error": "设备列表探测超时。",
            }

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            err = stderr if stderr else f"子进程异常退出 (code={completed.returncode})"
            return {
                "backend_ok": False,
                "device_count": 0,
                "devices": [],
                "error": err,
            }

        stdout = completed.stdout.strip().splitlines()
        if not stdout:
            return {
                "backend_ok": False,
                "device_count": 0,
                "devices": [],
                "error": "设备列表探测未返回结果。",
            }

        try:
            payload = json.loads(stdout[-1])
            devices = payload.get("devices", [])
            if not isinstance(devices, list):
                devices = []
            return {
                "backend_ok": True,
                "device_count": int(payload.get("device_count", 0)),
                "devices": devices,
                "error": "",
            }
        except Exception as exc:
            return {
                "backend_ok": False,
                "device_count": 0,
                "devices": [],
                "error": f"解析设备列表失败: {exc}",
            }

    @staticmethod
    def probe_stream_start_safe(
        width: int,
        height: int,
        fps: int,
        serial_number: str = "",
        timeout_sec: float = 8.0,
    ) -> Dict:
        """
        Try starting/stopping pipeline in subprocess to guard GUI from native crash.
        """
        probe_code = f"""
import json
import pyrealsense2 as rs

result = {{"ok": False, "error": ""}}

try:
    p = rs.pipeline()
    c = rs.config()
    if {bool(serial_number)!r}:
        c.enable_device({serial_number!r})
    c.enable_stream(rs.stream.color, {int(width)}, {int(height)}, rs.format.bgr8, {int(fps)})
    c.enable_stream(rs.stream.depth, {int(width)}, {int(height)}, rs.format.z16, {int(fps)})
    p.start(c)
    p.stop()
    result["ok"] = True
except Exception as e:
    result["error"] = repr(e)

print(json.dumps(result, ensure_ascii=False))
"""
        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": "相机流启动预检超时。",
                "crashed": False,
                "returncode": None,
            }

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            err = stderr if stderr else f"子进程异常退出 (code={completed.returncode})"
            return {
                "ok": False,
                "error": err,
                "crashed": completed.returncode < 0,
                "returncode": completed.returncode,
            }

        stdout = completed.stdout.strip().splitlines()
        if not stdout:
            return {
                "ok": False,
                "error": "预检未返回结果。",
                "crashed": False,
                "returncode": completed.returncode,
            }

        try:
            payload = json.loads(stdout[-1])
            ok = bool(payload.get("ok", False))
            return {
                "ok": ok,
                "error": str(payload.get("error", "")),
                "crashed": False,
                "returncode": completed.returncode,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"解析预检结果失败: {exc}",
                "crashed": False,
                "returncode": completed.returncode,
            }

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, cam_cfg: CameraConfig) -> None:
        if self._running:
            return

        self._ensure_rs_available()

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if cam_cfg.serial_number:
            self.config.enable_device(cam_cfg.serial_number)

        self.config.enable_stream(rs.stream.color, cam_cfg.width, cam_cfg.height, rs.format.bgr8, cam_cfg.fps)
        self.config.enable_stream(rs.stream.depth, cam_cfg.width, cam_cfg.height, rs.format.z16, cam_cfg.fps)

        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as exc:
            raise RuntimeError(f"相机启动失败: {exc}") from exc

        self.align = rs.align(rs.stream.color)

        device = self.profile.get_device()
        try:
            self.serial_number = device.get_info(rs.camera_info.serial_number)
        except Exception:
            self.serial_number = ""

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()

        self.intrinsics = {
            "width": int(color_intr.width),
            "height": int(color_intr.height),
            "fx": float(color_intr.fx),
            "fy": float(color_intr.fy),
            "cx": float(color_intr.ppx),
            "cy": float(color_intr.ppy),
            "distortion_model": str(color_intr.model),
            "distortion_coeffs": [float(v) for v in color_intr.coeffs],
            "depth_scale": float(self.depth_scale),
            "serial_number": self.serial_number,
        }

        self._running = True

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass

        self._running = False
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None

    def get_aligned_frames(self, timeout_ms: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned color/depth frames where depth is aligned to color."""
        if not self._running or self.pipeline is None or self.align is None:
            raise RuntimeError("相机未启动。")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except Exception as exc:
            raise RuntimeError(f"获取帧失败: {exc}") from exc

        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("帧数据不完整（color/depth 缺失）。")

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        if color is None or depth is None:
            raise RuntimeError("无法将帧转换为 numpy 数组。")

        return color, depth

    def get_intrinsics(self) -> Dict:
        if self.intrinsics is None:
            raise RuntimeError("内参不可用，请先启动相机。")
        return dict(self.intrinsics)
