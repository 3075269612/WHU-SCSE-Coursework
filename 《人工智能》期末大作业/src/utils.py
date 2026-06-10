from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    """将配置中的相对路径解析到项目根目录。"""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def ensure_directories(*paths: str | Path) -> None:
    for path in paths:
        project_path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """固定所有主要随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def file_nonempty(path: str | Path) -> bool:
    resolved = project_path(path)
    return resolved.exists() and resolved.is_file() and resolved.stat().st_size > 0

