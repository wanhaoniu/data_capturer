from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def pose_xyzrpy_to_matrix(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    degrees: bool = True,
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz + rpy (xyz order)."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()
    transform[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return transform


def pose_xyzrotvec_to_matrix(
    x: float,
    y: float,
    z: float,
    rx: float,
    ry: float,
    rz: float,
) -> np.ndarray:
    """Build a 4x4 homogeneous transform from xyz + rotation vector."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    transform[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return transform


def matrix_to_xyzrpy(transform: np.ndarray, degrees: bool = True) -> Tuple[float, float, float, float, float, float]:
    """Convert a 4x4 transform to xyz + rpy."""
    transform = np.asarray(transform, dtype=np.float64)
    x, y, z = transform[:3, 3]
    roll, pitch, yaw = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=degrees)
    return float(x), float(y), float(z), float(roll), float(pitch), float(yaw)


def matrix_to_xyzrotvec(transform: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Convert a 4x4 transform to xyz + rotation vector."""
    transform = np.asarray(transform, dtype=np.float64)
    x, y, z = transform[:3, 3]
    rx, ry, rz = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return float(x), float(y), float(z), float(rx), float(ry), float(rz)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    transform = np.asarray(transform, dtype=np.float64)
    inv = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def compose_transforms(*transforms: np.ndarray) -> np.ndarray:
    """Multiply a sequence of transforms from left to right."""
    result = np.eye(4, dtype=np.float64)
    for transform in transforms:
        result = result @ np.asarray(transform, dtype=np.float64)
    return result


def rtvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation/translation vectors to a 4x4 transform."""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def matrix_to_rtvec(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 4x4 transform to Rodrigues rotation and translation vectors."""
    transform = np.asarray(transform, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(3, 1)
    return rvec.reshape(3, 1), tvec


def matrix_to_quaternion(transform: np.ndarray) -> np.ndarray:
    """Return quaternion as [x, y, z, w] from a 4x4 transform."""
    transform = np.asarray(transform, dtype=np.float64)
    return Rotation.from_matrix(transform[:3, :3]).as_quat()


def average_transforms(transforms: Iterable[np.ndarray]) -> np.ndarray:
    """Compute mean transform using mean translation and averaged quaternion."""
    transform_list = [np.asarray(t, dtype=np.float64) for t in transforms]
    if not transform_list:
        raise ValueError("No transforms provided for averaging.")

    mean_translation = np.mean([t[:3, 3] for t in transform_list], axis=0)
    quaternions = np.array([Rotation.from_matrix(t[:3, :3]).as_quat() for t in transform_list])

    # Resolve quaternion sign ambiguity before averaging.
    reference = quaternions[0]
    for idx in range(len(quaternions)):
        if np.dot(reference, quaternions[idx]) < 0:
            quaternions[idx] *= -1.0

    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)

    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, :3] = Rotation.from_quat(mean_quaternion).as_matrix()
    mean_transform[:3, 3] = mean_translation
    return mean_transform


def rotation_error_deg(reference_rotation: np.ndarray, test_rotation: np.ndarray) -> float:
    """Angular distance between two 3x3 rotations in degrees."""
    reference_rotation = np.asarray(reference_rotation, dtype=np.float64)
    test_rotation = np.asarray(test_rotation, dtype=np.float64)
    relative = reference_rotation.T @ test_rotation
    angle_rad = Rotation.from_matrix(relative).magnitude()
    return float(np.degrees(angle_rad))


def transform_to_serializable(transform: np.ndarray) -> List[List[float]]:
    """Convert ndarray transform to nested Python list for JSON/YAML."""
    return np.asarray(transform, dtype=np.float64).tolist()
