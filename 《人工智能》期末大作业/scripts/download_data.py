from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.utils import PROJECT_ROOT, project_path


URLS = [
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
]


def download_file(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in URLS:
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, output_path)
            print(f"Saved: {output_path}")
            return
        except Exception as exc:
            last_error = exc
            print(f"Failed: {url} ({exc})")
    raise RuntimeError(f"所有下载地址均失败: {last_error}")


def extract_and_convert(zip_path: Path, raw_file: Path, csv_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_file.parent)
    if not raw_file.exists():
        raise FileNotFoundError(f"压缩包中没有找到目标文件: {raw_file}")

    rows = []
    with raw_file.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            label, text = line.rstrip("\n").split("\t", 1)
            rows.append({"label": label, "text": text})
    df = pd.DataFrame(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Converted CSV: {csv_path} ({len(df)} rows)")


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    zip_path = project_path(config["data"]["raw_zip"])
    raw_file = project_path(config["data"]["raw_file"])
    csv_path = project_path(config["data"]["dataset_csv"])

    if not zip_path.exists():
        download_file(zip_path)
    else:
        print(f"Using existing zip: {zip_path}")
    extract_and_convert(zip_path, raw_file, csv_path)


if __name__ == "__main__":
    main()

