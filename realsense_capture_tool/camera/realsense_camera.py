from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import cv2
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
    use_default_profile: bool = False


class RealSenseCamera:
    """Intel RealSense D435 wrapper with explicit depth-to-color alignment."""
    _shared_context = None

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
        context = RealSenseCamera._get_shared_context()
        return len(context.query_devices()) > 0

    @staticmethod
    def _get_shared_context():
        RealSenseCamera._ensure_rs_available()
        if RealSenseCamera._shared_context is None:
            RealSenseCamera._shared_context = rs.context()
        return RealSenseCamera._shared_context

    @staticmethod
    def first_device_info() -> Dict[str, str]:
        RealSenseCamera._ensure_rs_available()
        context = RealSenseCamera._get_shared_context()
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
    entry = {"name": "", "serial_number": "", "firmware_version": "", "usb_type": "", "product_line": ""}
    try:
        dev = devices[i]
        for key, field in [
            (rs.camera_info.name, "name"),
            (rs.camera_info.serial_number, "serial_number"),
            (rs.camera_info.firmware_version, "firmware_version"),
            (rs.camera_info.usb_type_descriptor, "usb_type"),
            (rs.camera_info.product_line, "product_line"),
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
        use_default_profile: bool = False,
        timeout_sec: float = 8.0,
    ) -> Dict:
        """
        Try starting/stopping pipeline in subprocess to guard GUI from native crash.
        """
        stream_setup = """
    c.enable_stream(rs.stream.color)
    c.enable_stream(rs.stream.depth)
""" if use_default_profile else f"""
    c.enable_stream(rs.stream.color, {int(width)}, {int(height)}, rs.format.any, {int(fps)})
    c.enable_stream(rs.stream.depth, {int(width)}, {int(height)}, rs.format.z16, {int(fps)})
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
{stream_setup}
    p.start(c)
    frames = p.wait_for_frames(3000)
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        raise RuntimeError("未获取到完整的 color/depth 帧。")
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

    @staticmethod
    def probe_multi_stream_start_safe(
        serial_numbers: List[str],
        width: int,
        height: int,
        fps: int,
        use_default_profile: bool = False,
        timeout_sec: float = 20.0,
        startup_delay_sec: float = 0.35,
    ) -> Dict:
        """
        Start multiple pipelines sequentially in a subprocess and wait for frames from all devices.
        """
        serials = [str(item).strip() for item in serial_numbers if str(item).strip()]
        stream_setup = """
        c.enable_stream(rs.stream.color)
        c.enable_stream(rs.stream.depth)
""" if use_default_profile else f"""
        c.enable_stream(rs.stream.color, {int(width)}, {int(height)}, rs.format.any, {int(fps)})
        c.enable_stream(rs.stream.depth, {int(width)}, {int(height)}, rs.format.z16, {int(fps)})
"""
        probe_code = f"""
import json
import time
import pyrealsense2 as rs

serials = {serials!r}
result = {{
    "ok": False,
    "error": "",
    "failed_serial": "",
    "stage": "",
    "started": [],
    "framed": [],
}}
pipes = []

try:
    for serial in serials:
        result["stage"] = "start"
        result["failed_serial"] = serial

        p = rs.pipeline()
        c = rs.config()
        c.enable_device(serial)
{stream_setup}
        p.start(c)
        pipes.append((serial, p))
        result["started"].append(serial)
        time.sleep({float(startup_delay_sec)!r})

    for serial, p in pipes:
        result["stage"] = "frames"
        result["failed_serial"] = serial
        frames = p.wait_for_frames(4000)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("未获取到完整的 color/depth 帧。")
        result["framed"].append(serial)

    result["ok"] = True
    result["failed_serial"] = ""
    result["stage"] = "done"
except Exception as e:
    result["error"] = repr(e)
finally:
    for _, p in pipes:
        try:
            p.stop()
        except Exception:
            pass

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
                "error": "多机联合预检超时。",
                "failed_serial": "",
                "stage": "timeout",
                "started": [],
                "framed": [],
                "crashed": False,
                "returncode": None,
            }

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            err = stderr if stderr else f"子进程异常退出 (code={completed.returncode})"
            return {
                "ok": False,
                "error": err,
                "failed_serial": "",
                "stage": "subprocess",
                "started": [],
                "framed": [],
                "crashed": completed.returncode < 0,
                "returncode": completed.returncode,
            }

        stdout = completed.stdout.strip().splitlines()
        if not stdout:
            return {
                "ok": False,
                "error": "多机联合预检未返回结果。",
                "failed_serial": "",
                "stage": "parse",
                "started": [],
                "framed": [],
                "crashed": False,
                "returncode": completed.returncode,
            }

        try:
            payload = json.loads(stdout[-1])
            return {
                "ok": bool(payload.get("ok", False)),
                "error": str(payload.get("error", "")),
                "failed_serial": str(payload.get("failed_serial", "")),
                "stage": str(payload.get("stage", "")),
                "started": payload.get("started", []),
                "framed": payload.get("framed", []),
                "crashed": False,
                "returncode": completed.returncode,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": f"解析多机预检结果失败: {exc}",
                "failed_serial": "",
                "stage": "parse",
                "started": [],
                "framed": [],
                "crashed": False,
                "returncode": completed.returncode,
            }

    @staticmethod
    def estimate_color_depth_stream_upper_bound_mbps(width: int, height: int, fps: int) -> float:
        """
        Rough upper-bound estimate using app-requested formats: color BGR8 + depth Z16.
        """
        pixels_per_second = int(width) * int(height) * int(fps)
        bits_per_pixel = 24 + 16
        return float((pixels_per_second * bits_per_pixel) / 1_000_000.0)

    @property
    def is_running(self) -> bool:
        return self._running

    @staticmethod
    def _reshape_packed_color(raw: np.ndarray, height: int, width: int, channels: int) -> np.ndarray:
        if raw.ndim == 3:
            return raw
        if raw.ndim == 2 and raw.shape == (height, width * channels):
            return raw.reshape(height, width, channels)
        if raw.ndim == 1 and raw.size == height * width * channels:
            return raw.reshape(height, width, channels)
        return raw

    @staticmethod
    def _color_frame_to_bgr(color_frame) -> np.ndarray:
        profile = color_frame.get_profile().as_video_stream_profile()
        fmt = profile.format()
        intr = profile.get_intrinsics()
        width = int(intr.width)
        height = int(intr.height)
        raw = np.asanyarray(color_frame.get_data())

        if fmt == rs.format.bgr8:
            color = raw
        elif fmt == rs.format.rgb8:
            color = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        elif fmt == rs.format.yuyv:
            yuyv = RealSenseCamera._reshape_packed_color(raw, height, width, 2)
            color = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        elif hasattr(rs.format, "uyvy") and fmt == rs.format.uyvy:
            uyvy = RealSenseCamera._reshape_packed_color(raw, height, width, 2)
            color = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
        elif fmt == rs.format.mjpeg:
            mjpeg = raw.reshape(-1)
            color = cv2.imdecode(mjpeg, cv2.IMREAD_COLOR)
            if color is None:
                raise RuntimeError("MJPEG 解码失败。")
        else:
            raise RuntimeError(f"不支持的彩色流格式: {fmt}")

        if color is None or color.size == 0:
            raise RuntimeError("彩色帧解码失败。")
        return color

    def start(self, cam_cfg: CameraConfig) -> None:
        if self._running:
            return

        self._ensure_rs_available()

        try:
            self.pipeline = rs.pipeline(self._get_shared_context())
        except TypeError:
            self.pipeline = rs.pipeline()
        self.config = rs.config()

        if cam_cfg.serial_number:
            self.config.enable_device(cam_cfg.serial_number)

        if cam_cfg.use_default_profile:
            self.config.enable_stream(rs.stream.color)
            self.config.enable_stream(rs.stream.depth)
        else:
            # Let librealsense negotiate the actual color format, matching the viewer/multicam sample behavior.
            self.config.enable_stream(rs.stream.color, cam_cfg.width, cam_cfg.height, rs.format.any, cam_cfg.fps)
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
        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intr = depth_stream.get_intrinsics()

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
            "color_format": str(color_stream.format()),
            "color_fps": int(color_stream.fps()),
            "depth_format": str(depth_stream.format()),
            "depth_width": int(depth_intr.width),
            "depth_height": int(depth_intr.height),
            "depth_fps": int(depth_stream.fps()),
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

        color = self._color_frame_to_bgr(color_frame)
        depth = np.asanyarray(depth_frame.get_data())

        if color is None or depth is None:
            raise RuntimeError("无法将帧转换为 numpy 数组。")

        return color, depth

    def get_intrinsics(self) -> Dict:
        if self.intrinsics is None:
            raise RuntimeError("内参不可用，请先启动相机。")
        return dict(self.intrinsics)
