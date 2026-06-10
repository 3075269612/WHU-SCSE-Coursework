from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    """解析本项目使用的轻量 YAML 标量，避免额外依赖 PyYAML。"""
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    """读取仅包含两层键值的项目配置文件。"""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    config: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            config[current_section] = {}
            continue
        if current_section is None or ":" not in line:
            raise ValueError(f"无法解析配置行: {raw_line}")
        key, value = line.strip().split(":", 1)
        config[current_section][key.strip()] = _parse_scalar(value)
    return config

