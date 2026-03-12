from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.io_utils import ensure_dir, read_json, write_json


@dataclass
class Sample:
    sample_id: int
    timestamp: str
    robot_pose: List[List[float]]
    target_pose_cam: List[List[float]]
    rvec: List[float]
    tvec: List[float]
    reproj_error: float
    image_filename: str = ""
    image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": int(self.sample_id),
            "timestamp": self.timestamp,
            "robot_pose": self.robot_pose,
            "target_pose_cam": self.target_pose_cam,
            "rvec": self.rvec,
            "tvec": self.tvec,
            "reproj_error": float(self.reproj_error),
            "image_filename": self.image_filename,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Sample":
        return Sample(
            sample_id=int(data["sample_id"]),
            timestamp=str(data["timestamp"]),
            robot_pose=data["robot_pose"],
            target_pose_cam=data["target_pose_cam"],
            rvec=[float(v) for v in data["rvec"]],
            tvec=[float(v) for v in data["tvec"]],
            reproj_error=float(data.get("reproj_error", 0.0)),
            image_filename=str(data.get("image_filename", "")),
            image=None,
        )


class SampleManager:
    def __init__(self) -> None:
        self.samples: List[Sample] = []
        self._next_id = 1

    def add_sample(
        self,
        image: np.ndarray,
        robot_pose: np.ndarray,
        target_pose_cam: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        reproj_error: float,
    ) -> Sample:
        sample = Sample(
            sample_id=self._next_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            robot_pose=np.asarray(robot_pose, dtype=np.float64).tolist(),
            target_pose_cam=np.asarray(target_pose_cam, dtype=np.float64).tolist(),
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3).tolist(),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3).tolist(),
            reproj_error=float(reproj_error),
            image_filename=f"images/sample_{self._next_id:04d}.png",
            image=image.copy(),
        )
        self.samples.append(sample)
        self._next_id += 1
        return sample

    def remove_sample(self, index: int) -> bool:
        if index < 0 or index >= len(self.samples):
            return False
        self.samples.pop(index)
        return True

    def clear(self) -> None:
        self.samples.clear()
        self._next_id = 1

    def get_samples(self) -> List[Sample]:
        return list(self.samples)

    def save_samples(self, project_dir: str | Path) -> Path:
        project_path = ensure_dir(project_dir)
        image_dir = ensure_dir(project_path / "images")

        for sample in self.samples:
            image_name = Path(sample.image_filename).name if sample.image_filename else f"sample_{sample.sample_id:04d}.png"
            sample.image_filename = str(Path("images") / image_name)
            image_path = image_dir / image_name
            if sample.image is not None:
                cv2.imwrite(str(image_path), sample.image)

        samples_path = project_path / "samples.json"
        write_json(samples_path, [sample.to_dict() for sample in self.samples])
        return samples_path

    def load_samples(self, project_dir: str | Path) -> List[Sample]:
        project_path = Path(project_dir)
        samples_data = read_json(project_path / "samples.json", default=[])

        self.samples = []
        max_id = 0
        for item in samples_data:
            sample = Sample.from_dict(item)
            if sample.image_filename:
                image_path = project_path / sample.image_filename
                if image_path.exists():
                    sample.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            self.samples.append(sample)
            max_id = max(max_id, sample.sample_id)

        self._next_id = max_id + 1
        return self.get_samples()
