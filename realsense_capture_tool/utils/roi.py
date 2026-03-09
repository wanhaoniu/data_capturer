from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    def is_valid(self) -> bool:
        return self.w > 0 and self.h > 0 and self.x >= 0 and self.y >= 0

    def clamp(self, image_shape: Tuple[int, int]) -> "ROI":
        h_img, w_img = image_shape[:2]

        x = max(0, min(self.x, w_img - 1))
        y = max(0, min(self.y, h_img - 1))

        max_w = w_img - x
        max_h = h_img - y

        w = max(0, min(self.w, max_w))
        h = max(0, min(self.h, max_h))

        return ROI(x=x, y=y, w=w, h=h)

    def to_dict(self) -> dict:
        return {"roi_x": int(self.x), "roi_y": int(self.y), "roi_w": int(self.w), "roi_h": int(self.h)}


def select_roi_with_opencv(image_bgr: np.ndarray, window_name: str = "Select ROI") -> Optional[ROI]:
    if image_bgr is None or image_bgr.size == 0:
        return None

    display = image_bgr.copy()
    rect = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = [int(v) for v in rect]
    roi = ROI(x=x, y=y, w=w, h=h)

    if not roi.is_valid():
        return None
    return roi


def crop_with_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    if image is None:
        raise ValueError("输入图像为空。")

    clamped = roi.clamp(image.shape[:2])
    if not clamped.is_valid():
        raise ValueError("ROI 非法或超出图像范围。")

    return image[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w]
