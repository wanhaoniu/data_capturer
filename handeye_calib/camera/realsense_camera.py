from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except Exception:
    rs = None
    REALSENSE_AVAILABLE = False


class RealSenseCamera:
    """Thin wrapper around Intel RealSense color stream."""

    def __init__(self) -> None:
        self.pipeline = None
        self.profile = None
        self.is_open = False
        self._intrinsics: Optional[Dict[str, object]] = None

    @property
    def available(self) -> bool:
        return REALSENSE_AVAILABLE

    def open(self, width: int = 1280, height: int = 720, fps: int = 30) -> Tuple[bool, str]:
        if not REALSENSE_AVAILABLE:
            return False, "pyrealsense2 未安装或不可用。"

        if self.is_open:
            return True, "相机已打开。"

        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            self.profile = self.pipeline.start(config)

            stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
            self._intrinsics = {
                "width": int(intr.width),
                "height": int(intr.height),
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "model": str(intr.model),
                "coeffs": [float(c) for c in intr.coeffs],
            }
            self.is_open = True
            return True, "相机打开成功。"
        except Exception as exc:
            self.close()
            return False, f"打开相机失败: {exc}"

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None
        self.profile = None
        self.is_open = False

    def get_frame(self) -> Tuple[Optional[np.ndarray], str]:
        if not self.is_open or self.pipeline is None:
            return None, "相机未打开。"

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None, "未获取到彩色帧。"
            image = np.asanyarray(color_frame.get_data()).copy()
            return image, ""
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "device" in lowered and ("disconnect" in lowered or "no device" in lowered):
                self.close()
            return None, f"读取帧失败: {message}"

    def get_intrinsics_dict(self) -> Dict[str, object]:
        return dict(self._intrinsics) if self._intrinsics is not None else {}

    def get_camera_matrix(self) -> Optional[np.ndarray]:
        if not self._intrinsics:
            return None
        return np.array(
            [
                [self._intrinsics["fx"], 0.0, self._intrinsics["ppx"]],
                [0.0, self._intrinsics["fy"], self._intrinsics["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_dist_coeffs(self) -> Optional[np.ndarray]:
        if not self._intrinsics:
            return None
        coeffs = np.array(self._intrinsics.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
        if coeffs.size < 5:
            coeffs = np.pad(coeffs, (0, 5 - coeffs.size), mode="constant")
        return coeffs.reshape(-1, 1)
