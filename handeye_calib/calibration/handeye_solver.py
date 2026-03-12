from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.transforms import (
    average_transforms,
    compose_transforms,
    invert_transform,
    rotation_error_deg,
)
from data.sample_manager import Sample


METHOD_MAP: Dict[str, int] = {
    "Tsai": cv2.CALIB_HAND_EYE_TSAI,
    "Park": cv2.CALIB_HAND_EYE_PARK,
    "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

MODE_EYE_IN_HAND = "eye_in_hand"
MODE_EYE_TO_HAND = "eye_to_hand"


@dataclass
class HandEyeResult:
    mode: str
    method: str
    transform_name: str
    transform_matrix: np.ndarray
    translation: np.ndarray
    rotation_matrix: np.ndarray
    euler_deg: np.ndarray
    quaternion_xyzw: np.ndarray
    per_sample_errors: List[Dict[str, float]]
    mean_translation_error_m: float
    max_translation_error_m: float
    mean_rotation_error_deg: float
    max_rotation_error_deg: float
    consistency_transform_name: str
    consistency_mean_transform: np.ndarray
    derived_transforms: Dict[str, List[List[float]]]
    created_at: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "method": self.method,
            "transform_name": self.transform_name,
            "transform_matrix": self.transform_matrix.tolist(),
            "translation": self.translation.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "euler_deg_xyz": self.euler_deg.tolist(),
            "quaternion_xyzw": self.quaternion_xyzw.tolist(),
            "per_sample_errors": self.per_sample_errors,
            "mean_translation_error_m": self.mean_translation_error_m,
            "max_translation_error_m": self.max_translation_error_m,
            "mean_rotation_error_deg": self.mean_rotation_error_deg,
            "max_rotation_error_deg": self.max_rotation_error_deg,
            "consistency_transform_name": self.consistency_transform_name,
            "consistency_mean_transform": self.consistency_mean_transform.tolist(),
            "derived_transforms": self.derived_transforms,
            "created_at": self.created_at,
        }


class HandEyeSolver:
    def solve(self, samples: List[Sample], mode: str, method: str) -> HandEyeResult:
        if len(samples) < 3:
            raise ValueError("手眼标定至少需要 3 个样本。")

        if method not in METHOD_MAP:
            raise ValueError(f"不支持的标定方法: {method}")

        if mode not in {MODE_EYE_IN_HAND, MODE_EYE_TO_HAND}:
            raise ValueError(f"不支持的标定模式: {mode}")

        rotations_gripper2base: List[np.ndarray] = []
        translations_gripper2base: List[np.ndarray] = []
        rotations_target2cam: List[np.ndarray] = []
        translations_target2cam: List[np.ndarray] = []

        for sample in samples:
            base_tool = np.asarray(sample.robot_pose, dtype=np.float64)
            cam_target = np.asarray(sample.target_pose_cam, dtype=np.float64)

            if mode == MODE_EYE_IN_HAND:
                gripper2base = base_tool
            else:
                # Eye-to-Hand: map to OpenCV hand-eye form with toolT_base as gripper2base.
                gripper2base = invert_transform(base_tool)

            rotations_gripper2base.append(gripper2base[:3, :3])
            translations_gripper2base.append(gripper2base[:3, 3].reshape(3, 1))
            rotations_target2cam.append(cam_target[:3, :3])
            translations_target2cam.append(cam_target[:3, 3].reshape(3, 1))

        rotation_cam2gripper, translation_cam2gripper = cv2.calibrateHandEye(
            rotations_gripper2base,
            translations_gripper2base,
            rotations_target2cam,
            translations_target2cam,
            method=METHOD_MAP[method],
        )

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation_cam2gripper, dtype=np.float64)
        transform[:3, 3] = np.asarray(translation_cam2gripper, dtype=np.float64).reshape(3)

        if mode == MODE_EYE_IN_HAND:
            transform_name = "T_tool_camera"
            consistency_transform_name = "T_base_target"
            derived_transforms = {
                "T_camera_tool": invert_transform(transform).tolist(),
            }
            consistency_transforms = [
                compose_transforms(
                    np.asarray(sample.robot_pose, dtype=np.float64),
                    transform,
                    np.asarray(sample.target_pose_cam, dtype=np.float64),
                )
                for sample in samples
            ]
        else:
            transform_name = "T_base_camera"
            consistency_transform_name = "T_tool_target"
            derived_transforms = {
                "T_camera_base": invert_transform(transform).tolist(),
            }
            consistency_transforms = [
                compose_transforms(
                    invert_transform(np.asarray(sample.robot_pose, dtype=np.float64)),
                    transform,
                    np.asarray(sample.target_pose_cam, dtype=np.float64),
                )
                for sample in samples
            ]

        consistency_mean = average_transforms(consistency_transforms)
        per_sample_errors: List[Dict[str, float]] = []
        t_errors = []
        r_errors = []

        for sample, consistency_transform in zip(samples, consistency_transforms):
            t_error = float(np.linalg.norm(consistency_transform[:3, 3] - consistency_mean[:3, 3]))
            r_error = rotation_error_deg(consistency_mean[:3, :3], consistency_transform[:3, :3])
            t_errors.append(t_error)
            r_errors.append(r_error)
            per_sample_errors.append(
                {
                    "sample_id": int(sample.sample_id),
                    "translation_error_m": t_error,
                    "rotation_error_deg": float(r_error),
                    "reprojection_error_px": float(sample.reproj_error),
                }
            )

        rotation_matrix = transform[:3, :3]
        translation = transform[:3, 3]
        euler_deg = Rotation.from_matrix(rotation_matrix).as_euler("xyz", degrees=True)
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        return HandEyeResult(
            mode=mode,
            method=method,
            transform_name=transform_name,
            transform_matrix=transform,
            translation=translation,
            rotation_matrix=rotation_matrix,
            euler_deg=euler_deg,
            quaternion_xyzw=quaternion,
            per_sample_errors=per_sample_errors,
            mean_translation_error_m=float(np.mean(t_errors)),
            max_translation_error_m=float(np.max(t_errors)),
            mean_rotation_error_deg=float(np.mean(r_errors)),
            max_rotation_error_deg=float(np.max(r_errors)),
            consistency_transform_name=consistency_transform_name,
            consistency_mean_transform=consistency_mean,
            derived_transforms=derived_transforms,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
