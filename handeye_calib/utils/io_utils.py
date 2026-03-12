from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.write_text(json.dumps(_to_builtin(data), indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def write_yaml(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.write_text(yaml.safe_dump(_to_builtin(data), sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_matrix_txt(path: str | Path, matrix: np.ndarray, name: str = "T") -> None:
    target = Path(path)
    matrix = np.asarray(matrix, dtype=np.float64)

    lines = [f"{name} (4x4):"]
    for row in matrix:
        lines.append(" ".join(f"{val: .8f}" for val in row))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
