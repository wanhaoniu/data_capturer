from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


CSV_FIELDS = [
    "sample_id",
    "sample_name",
    "light_label",
    "left_rgb_path",
    "left_depth_path",
    "left_depth_vis_path",
    "left_normals_path",
    "left_width",
    "left_height",
    "middle_rgb_path",
    "middle_depth_path",
    "middle_depth_vis_path",
    "middle_normals_path",
    "middle_width",
    "middle_height",
    "right_rgb_path",
    "right_depth_path",
    "right_depth_vis_path",
    "right_normals_path",
    "right_width",
    "right_height",
    "roi_enabled",
    "roi_x",
    "roi_y",
    "roi_w",
    "roi_h",
    "depth_aligned_to_color",
    "intrinsics_file",
    "saved_in_session",
]


class SessionMetadataManager:
    def __init__(self) -> None:
        self._records: List[Dict] = []
        self._dirty: bool = False
        self._last_export_at: Optional[str] = None

    @property
    def records(self) -> List[Dict]:
        return deepcopy(self._records)

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def has_unexported_changes(self) -> bool:
        return self._dirty

    @property
    def last_export_at(self) -> Optional[str]:
        return self._last_export_at

    def add_record(self, record: Dict) -> None:
        self._records.append(deepcopy(record))
        self._dirty = True

    def get_last_record(self) -> Optional[Dict]:
        if not self._records:
            return None
        return deepcopy(self._records[-1])

    def pop_last_record(self) -> Optional[Dict]:
        if not self._records:
            return None
        rec = self._records.pop()
        self._dirty = True
        return rec

    def clear(self) -> None:
        self._records.clear()
        self._dirty = False
        self._last_export_at = None

    def export(self, root_dir: str) -> Dict[str, Path]:
        if len(self._records) == 0:
            raise RuntimeError("当前会话没有任何样本可导出。")

        if not self._dirty:
            raise RuntimeError("当前没有新的 metadata 变更。")

        root = Path(root_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        metadata_path = root / "metadata.json"
        csv_path = root / "index.csv"

        exported_at = datetime.now().isoformat(timespec="seconds")
        payload = {
            "exported_at": exported_at,
            "total_samples": len(self._records),
            "samples": self._records,
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            raise RuntimeError(f"导出 metadata.json 失败: {exc}") from exc

        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for row in self._records:
                    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
        except Exception as exc:
            raise RuntimeError(f"导出 index.csv 失败: {exc}") from exc

        self._dirty = False
        self._last_export_at = exported_at

        return {"metadata_json": metadata_path, "index_csv": csv_path}
