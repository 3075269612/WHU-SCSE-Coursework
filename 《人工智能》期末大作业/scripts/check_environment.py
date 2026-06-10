from __future__ import annotations

import importlib
import importlib.metadata as metadata
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import PROJECT_ROOT


REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("docx", "python-docx"),
]

OPTIONAL = [
    ("tqdm", "tqdm"),
    ("pypdf", "pypdf"),
]


def package_status(module_name: str, package_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, metadata.version(package_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    expected_python = config["project"]["python"]
    print(f"当前解释器: {sys.executable}")
    print(f"计划解释器: {expected_python}")
    print(f"Python 版本: {sys.version.replace(chr(10), ' ')}")
    print("\n必需依赖:")
    failed = False
    for module_name, package_name in REQUIRED:
        ok, detail = package_status(module_name, package_name)
        print(f"  {package_name}: {'OK' if ok else 'MISSING'} {detail}")
        failed = failed or not ok
    print("\n可选依赖:")
    for module_name, package_name in OPTIONAL:
        ok, detail = package_status(module_name, package_name)
        print(f"  {package_name}: {'OK' if ok else 'MISSING'} {detail}")
    if failed:
        raise SystemExit("环境检查失败：缺少必需依赖。")
    print("\n环境检查通过。")


if __name__ == "__main__":
    main()

