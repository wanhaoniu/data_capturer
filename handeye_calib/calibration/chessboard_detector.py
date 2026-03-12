from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from calibration.transforms import rtvec_to_matrix


@dataclass
class DetectionResult:
    success: bool
    message: str
    visualization: np.ndarray
    corners: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    target_pose_cam: Optional[np.ndarray] = None
    reproj_error: float = 0.0


class ChessboardDetector:
    def __init__(self, columns: int = 9, rows: int = 6, square_size: float = 0.025) -> None:
        self.columns = int(columns)
        self.rows = int(rows)
        self.square_size = float(square_size)
        self._update_object_points()

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return self.columns, self.rows

    def set_pattern(self, columns: int, rows: int, square_size: float) -> None:
        self.columns = int(columns)
        self.rows = int(rows)
        self.square_size = float(square_size)
        self._update_object_points()

    def _update_object_points(self) -> None:
        grid = np.mgrid[0 : self.columns, 0 : self.rows].T.reshape(-1, 2)
        object_points = np.zeros((self.rows * self.columns, 3), dtype=np.float32)
        object_points[:, :2] = grid.astype(np.float32) * self.square_size
        self.object_points = object_points

    def detect(
        self,
        image_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> DetectionResult:
        if image_bgr is None:
            return DetectionResult(False, "图像为空。", visualization=np.zeros((480, 640, 3), dtype=np.uint8))

        visualization = image_bgr.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)

        if not found:
            return DetectionResult(False, "未检测到棋盘格。", visualization=visualization)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(visualization, self.pattern_size, corners_refined, found)

        ok, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners_refined,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return DetectionResult(False, "solvePnP 求解失败。", visualization=visualization)

        try:
            axis_length = max(self.square_size * 3.0, 0.02)
            cv2.drawFrameAxes(visualization, camera_matrix, dist_coeffs, rvec, tvec, axis_length, 2)
        except Exception:
            pass

        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, camera_matrix, dist_coeffs)
        pixel_error = np.linalg.norm(
            corners_refined.reshape(-1, 2) - projected.reshape(-1, 2),
            axis=1,
        )
        reproj_error = float(np.mean(pixel_error))

        target_pose_cam = rtvec_to_matrix(rvec, tvec)
        return DetectionResult(
            success=True,
            message="检测成功。",
            visualization=visualization,
            corners=corners_refined,
            rvec=rvec.reshape(3, 1),
            tvec=tvec.reshape(3, 1),
            target_pose_cam=target_pose_cam,
            reproj_error=reproj_error,
        )
