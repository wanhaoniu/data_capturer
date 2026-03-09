from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from utils.roi import ROI, crop_with_roi


SAMPLE_PATTERN = re.compile(r"^(\d{6})_.*_rgb\.png$")
CAMERA_KEYS = ("left", "middle", "right")


def depth_to_colormap(depth_u16: np.ndarray) -> np.ndarray:
    if depth_u16.ndim != 2:
        raise ValueError("depth 图必须是二维单通道。")

    depth = depth_u16.astype(np.float32)
    valid = depth > 0

    vis = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        p95 = np.percentile(depth[valid], 95)
        p95 = max(p95, 1.0)
        vis = np.clip((depth / p95) * 255.0, 0, 255).astype(np.uint8)

    color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    color[~valid] = 0
    return color


class DatasetSaver:
    def __init__(self, root_dir: str) -> None:
        self.root_dir: Path = Path(root_dir).expanduser().resolve()
        self.rgb_dir: Path = self.root_dir / "rgb"
        self.depth_dir: Path = self.root_dir / "depth"
        self.depth_vis_dir: Path = self.root_dir / "depth_vis"
        self.normals_dir: Path = self.root_dir / "normals"
        self.camera_intrinsics_file: Path = self.root_dir / "camera_intrinsics.json"

        self.max_sample_id: int = 0
        self.ensure_dataset_dirs()
        self.max_sample_id = self.scan_max_sample_id()

    def set_root_dir(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.rgb_dir = self.root_dir / "rgb"
        self.depth_dir = self.root_dir / "depth"
        self.depth_vis_dir = self.root_dir / "depth_vis"
        self.normals_dir = self.root_dir / "normals"
        self.camera_intrinsics_file = self.root_dir / "camera_intrinsics.json"

        self.ensure_dataset_dirs()
        self.max_sample_id = self.scan_max_sample_id()

    def ensure_dataset_dirs(self) -> None:
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.rgb_dir.mkdir(parents=True, exist_ok=True)
            self.depth_dir.mkdir(parents=True, exist_ok=True)
            self.depth_vis_dir.mkdir(parents=True, exist_ok=True)
            self.normals_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"创建保存目录失败: {exc}") from exc

        test_file = self.root_dir / ".write_test"
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("ok")
            test_file.unlink(missing_ok=True)
        except Exception as exc:
            raise RuntimeError(f"保存目录不可写: {self.root_dir} ({exc})") from exc

    def scan_max_sample_id(self) -> int:
        max_id = 0
        if not self.rgb_dir.exists():
            return max_id

        for file_name in os.listdir(self.rgb_dir):
            match = SAMPLE_PATTERN.match(file_name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return max_id

    def next_sample_id(self) -> int:
        return self.max_sample_id + 1

    @staticmethod
    def sanitize_light_label(label: str) -> str:
        clean = label.strip()
        if not clean:
            return "light"
        clean = re.sub(r"\s+", "_", clean)
        clean = re.sub(r"[^a-zA-Z0-9_-]", "", clean)
        return clean or "light"

    def save_intrinsics(self, intrinsics: Dict) -> Path:
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "intrinsics": intrinsics,
        }

        try:
            with open(self.camera_intrinsics_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise RuntimeError(f"写入相机内参失败: {exc}") from exc

        return self.camera_intrinsics_file

    def save_sample(
        self,
        color_bgr: np.ndarray,
        depth_u16: np.ndarray,
        normals_vis_bgr: np.ndarray,
        light_label: str,
        roi: Optional[ROI],
    ) -> Dict:
        if color_bgr is None or depth_u16 is None or normals_vis_bgr is None:
            raise RuntimeError("保存失败：输入帧为空。")

        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise RuntimeError("保存失败：RGB 帧格式非法。")

        if depth_u16.ndim != 2:
            raise RuntimeError("保存失败：Depth 帧格式非法。")

        if normals_vis_bgr.ndim != 3 or normals_vis_bgr.shape[2] != 3:
            raise RuntimeError("保存失败：Normals 帧格式非法。")

        if color_bgr.shape[:2] != depth_u16.shape[:2] or color_bgr.shape[:2] != normals_vis_bgr.shape[:2]:
            raise RuntimeError("保存失败：RGB/Depth/Normals 尺寸不一致。")

        roi_enabled = roi is not None
        roi_record = {"roi_x": 0, "roi_y": 0, "roi_w": 0, "roi_h": 0}

        color_to_save = color_bgr
        depth_to_save = depth_u16
        normals_to_save = normals_vis_bgr

        if roi is not None:
            clamped = roi.clamp(color_bgr.shape[:2])
            if not clamped.is_valid():
                raise RuntimeError("保存失败：ROI 非法。")

            color_to_save = crop_with_roi(color_bgr, clamped)
            depth_to_save = crop_with_roi(depth_u16, clamped)
            normals_to_save = crop_with_roi(normals_vis_bgr, clamped)
            roi_record = clamped.to_dict()

        sample_id = self.next_sample_id()
        clean_label = self.sanitize_light_label(light_label)
        sample_name = f"{sample_id:06d}_{clean_label}"

        rgb_rel = Path("rgb") / f"{sample_name}_rgb.png"
        depth_rel = Path("depth") / f"{sample_name}_depth.png"
        depth_vis_rel = Path("depth_vis") / f"{sample_name}_depth_vis.png"
        normals_rel = Path("normals") / f"{sample_name}_normals.png"

        rgb_abs = self.root_dir / rgb_rel
        depth_abs = self.root_dir / depth_rel
        depth_vis_abs = self.root_dir / depth_vis_rel
        normals_abs = self.root_dir / normals_rel

        depth_vis = depth_to_colormap(depth_to_save)

        written = []
        try:
            if not cv2.imwrite(str(rgb_abs), color_to_save):
                raise RuntimeError(f"写入文件失败: {rgb_abs}")
            written.append(rgb_abs)

            if not cv2.imwrite(str(depth_abs), depth_to_save):
                raise RuntimeError(f"写入文件失败: {depth_abs}")
            written.append(depth_abs)

            if not cv2.imwrite(str(depth_vis_abs), depth_vis):
                raise RuntimeError(f"写入文件失败: {depth_vis_abs}")
            written.append(depth_vis_abs)

            if not cv2.imwrite(str(normals_abs), normals_to_save):
                raise RuntimeError(f"写入文件失败: {normals_abs}")
            written.append(normals_abs)

        except Exception as exc:
            for path in written:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise RuntimeError(f"保存样本失败: {exc}") from exc

        self.max_sample_id = sample_id

        h, w = color_to_save.shape[:2]
        record = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "light_label": clean_label,
            "rgb_path": rgb_rel.as_posix(),
            "depth_path": depth_rel.as_posix(),
            "depth_vis_path": depth_vis_rel.as_posix(),
            "normals_path": normals_rel.as_posix(),
            "width": int(w),
            "height": int(h),
            "roi_enabled": bool(roi_enabled),
            "roi_x": int(roi_record["roi_x"]),
            "roi_y": int(roi_record["roi_y"]),
            "roi_w": int(roi_record["roi_w"]),
            "roi_h": int(roi_record["roi_h"]),
            "depth_aligned_to_color": True,
            "intrinsics_file": self.camera_intrinsics_file.name,
            "saved_in_session": True,
        }
        return record

    def save_multi_sample(
        self,
        camera_frames: Dict[str, Dict[str, np.ndarray]],
        light_label: str,
        roi: Optional[ROI] = None,
        roi_by_camera: Optional[Dict[str, ROI]] = None,
    ) -> Dict:
        missing = [k for k in CAMERA_KEYS if k not in camera_frames]
        if missing:
            raise RuntimeError(f"保存失败：缺少相机数据 {missing}")

        sample_id = self.next_sample_id()
        clean_label = self.sanitize_light_label(light_label)
        sample_name = f"{sample_id:06d}_{clean_label}"

        roi_map = dict(roi_by_camera or {})
        roi_enabled = (roi is not None) or (len(roi_map) > 0)
        zero_roi = {"roi_x": 0, "roi_y": 0, "roi_w": 0, "roi_h": 0}

        written = []
        record: Dict = {
            "sample_id": sample_id,
            "sample_name": sample_name,
            "light_label": clean_label,
            "roi_enabled": bool(roi_enabled),
            "roi_x": int(zero_roi["roi_x"]),
            "roi_y": int(zero_roi["roi_y"]),
            "roi_w": int(zero_roi["roi_w"]),
            "roi_h": int(zero_roi["roi_h"]),
            "depth_aligned_to_color": True,
            "intrinsics_file": self.camera_intrinsics_file.name,
            "saved_in_session": True,
        }
        for cam_key in CAMERA_KEYS:
            record[f"{cam_key}_roi_x"] = 0
            record[f"{cam_key}_roi_y"] = 0
            record[f"{cam_key}_roi_w"] = 0
            record[f"{cam_key}_roi_h"] = 0

        try:
            for cam_key in CAMERA_KEYS:
                entry = camera_frames[cam_key]
                color_bgr = entry.get("color")
                depth_u16 = entry.get("depth")
                normals_vis_bgr = entry.get("normals")

                if color_bgr is None or depth_u16 is None or normals_vis_bgr is None:
                    raise RuntimeError(f"{cam_key} 相机帧为空。")
                if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
                    raise RuntimeError(f"{cam_key} RGB 帧格式非法。")
                if depth_u16.ndim != 2:
                    raise RuntimeError(f"{cam_key} Depth 帧格式非法。")
                if normals_vis_bgr.ndim != 3 or normals_vis_bgr.shape[2] != 3:
                    raise RuntimeError(f"{cam_key} Normals 帧格式非法。")
                if color_bgr.shape[:2] != depth_u16.shape[:2] or color_bgr.shape[:2] != normals_vis_bgr.shape[:2]:
                    raise RuntimeError(f"{cam_key} RGB/Depth/Normals 尺寸不一致。")

                color_to_save = color_bgr
                depth_to_save = depth_u16
                normals_to_save = normals_vis_bgr
                cam_roi_record = dict(zero_roi)

                cam_roi = roi_map.get(cam_key)
                if cam_roi is None and roi is not None:
                    cam_roi = roi

                if cam_roi is not None:
                    clamped = cam_roi.clamp(color_bgr.shape[:2])
                    if not clamped.is_valid():
                        raise RuntimeError(f"{cam_key} ROI 非法。")
                    color_to_save = crop_with_roi(color_bgr, clamped)
                    depth_to_save = crop_with_roi(depth_u16, clamped)
                    normals_to_save = crop_with_roi(normals_vis_bgr, clamped)
                    cam_roi_record = clamped.to_dict()

                cam_sample_name = f"{sample_name}_{cam_key}"
                rgb_rel = Path("rgb") / f"{cam_sample_name}_rgb.png"
                depth_rel = Path("depth") / f"{cam_sample_name}_depth.png"
                depth_vis_rel = Path("depth_vis") / f"{cam_sample_name}_depth_vis.png"
                normals_rel = Path("normals") / f"{cam_sample_name}_normals.png"

                rgb_abs = self.root_dir / rgb_rel
                depth_abs = self.root_dir / depth_rel
                depth_vis_abs = self.root_dir / depth_vis_rel
                normals_abs = self.root_dir / normals_rel

                depth_vis = depth_to_colormap(depth_to_save)

                if not cv2.imwrite(str(rgb_abs), color_to_save):
                    raise RuntimeError(f"写入文件失败: {rgb_abs}")
                written.append(rgb_abs)
                if not cv2.imwrite(str(depth_abs), depth_to_save):
                    raise RuntimeError(f"写入文件失败: {depth_abs}")
                written.append(depth_abs)
                if not cv2.imwrite(str(depth_vis_abs), depth_vis):
                    raise RuntimeError(f"写入文件失败: {depth_vis_abs}")
                written.append(depth_vis_abs)
                if not cv2.imwrite(str(normals_abs), normals_to_save):
                    raise RuntimeError(f"写入文件失败: {normals_abs}")
                written.append(normals_abs)

                h, w = color_to_save.shape[:2]
                record[f"{cam_key}_rgb_path"] = rgb_rel.as_posix()
                record[f"{cam_key}_depth_path"] = depth_rel.as_posix()
                record[f"{cam_key}_depth_vis_path"] = depth_vis_rel.as_posix()
                record[f"{cam_key}_normals_path"] = normals_rel.as_posix()
                record[f"{cam_key}_width"] = int(w)
                record[f"{cam_key}_height"] = int(h)
                record[f"{cam_key}_roi_x"] = int(cam_roi_record["roi_x"])
                record[f"{cam_key}_roi_y"] = int(cam_roi_record["roi_y"])
                record[f"{cam_key}_roi_w"] = int(cam_roi_record["roi_w"])
                record[f"{cam_key}_roi_h"] = int(cam_roi_record["roi_h"])

            fallback = (
                record.get("middle_roi_x", 0),
                record.get("middle_roi_y", 0),
                record.get("middle_roi_w", 0),
                record.get("middle_roi_h", 0),
            )
            if fallback[2] <= 0 or fallback[3] <= 0:
                for cam_key in CAMERA_KEYS:
                    fw = int(record.get(f"{cam_key}_roi_w", 0))
                    fh = int(record.get(f"{cam_key}_roi_h", 0))
                    if fw > 0 and fh > 0:
                        fallback = (
                            int(record.get(f"{cam_key}_roi_x", 0)),
                            int(record.get(f"{cam_key}_roi_y", 0)),
                            fw,
                            fh,
                        )
                        break
            record["roi_x"], record["roi_y"], record["roi_w"], record["roi_h"] = fallback

        except Exception as exc:
            for path in written:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise RuntimeError(f"保存三路样本失败: {exc}") from exc

        self.max_sample_id = sample_id
        return record

    def delete_sample_files(self, record: Dict) -> None:
        for key, rel in record.items():
            if not key.endswith("_path"):
                continue
            if not rel:
                continue
            abs_path = self.root_dir / rel
            try:
                abs_path.unlink(missing_ok=True)
            except Exception as exc:
                raise RuntimeError(f"删除文件失败: {abs_path} ({exc})") from exc
