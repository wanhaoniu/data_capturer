from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_dict(cls, data: Dict) -> "CameraIntrinsics":
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
        )


class NormalsEstimator:
    """
    Estimate normals from aligned depth using camera intrinsics.

    Pipeline:
    1) depth(u16) -> metric depth(m)
    2) project to 3D points in camera coordinates
    3) compute local tangents by Sobel derivatives
    4) cross product + normalization
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intr = intrinsics

    @classmethod
    def from_intrinsics_dict(cls, intrinsics: Dict) -> "NormalsEstimator":
        return cls(CameraIntrinsics.from_dict(intrinsics))

    def compute_normals(self, depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
        if depth_u16.ndim != 2:
            raise ValueError("depth 图必须是单通道二维数组。")

        depth_m = depth_u16.astype(np.float32) * float(depth_scale)
        depth_m = cv2.GaussianBlur(depth_m, (5, 5), 0)

        h, w = depth_m.shape
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        z = depth_m
        valid = z > 1e-6

        x = (xx - self.intr.cx) * z / self.intr.fx
        y = (yy - self.intr.cy) * z / self.intr.fy

        # Use Sobel derivatives to get local tangent vectors in x/y directions.
        dx_x = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        dx_y = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        dx_z = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)

        dy_x = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        dy_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        dy_z = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)

        tangent_x = np.stack((dx_x, dx_y, dx_z), axis=-1)
        tangent_y = np.stack((dy_x, dy_y, dy_z), axis=-1)

        normals = np.cross(tangent_x, tangent_y)

        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.maximum(norm, 1e-12)

        invalid = (~valid) | (norm[..., 0] < 1e-9)
        normals[invalid] = 0.0

        # Keep normal orientation consistent: make Z non-negative for visualization stability.
        flip_mask = normals[..., 2] < 0
        normals[flip_mask] *= -1.0

        return normals.astype(np.float32)

    @staticmethod
    def normals_to_vis(normals: np.ndarray) -> np.ndarray:
        if normals.ndim != 3 or normals.shape[2] != 3:
            raise ValueError("normals 必须是 HxWx3 数组。")

        vis = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        return vis
