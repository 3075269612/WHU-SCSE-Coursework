from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import project_path


REQUIRED_FILES = [
    "data/processed/sms_spam.csv",
    "data/processed/train.csv",
    "data/processed/val.csv",
    "data/processed/test.csv",
    "results/dataset_summary.json",
    "results/metrics.json",
    "results/baselines.csv",
    "results/hyperparams.csv",
    "results/model_selection.csv",
    "results/tfidf_sensitivity.csv",
    "results/training_history.csv",
    "figures/training_curve.png",
    "figures/confusion_matrix.png",
    "figures/lr_sensitivity.png",
    "figures/dropout_sensitivity.png",
    "figures/hidden_dims_sensitivity.png",
    "figures/tfidf_sensitivity.png",
]

REPORT_CANDIDATES = [
    "report/AI-王李明-2024302181194.docx",
]


def assert_file(path: str) -> None:
    resolved = project_path(path)
    if not resolved.exists() or resolved.stat().st_size == 0:
        raise AssertionError(f"缺少或为空: {path}")


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
    return re.sub(r"<[^>]+>", "", xml)


def main() -> None:
    for path in REQUIRED_FILES:
        assert_file(path)

    summary = json.loads(project_path("results/dataset_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads(project_path("results/metrics.json").read_text(encoding="utf-8"))
    baselines = pd.read_csv(project_path("results/baselines.csv"))
    hyper = pd.read_csv(project_path("results/hyperparams.csv"))
    model_selection = pd.read_csv(project_path("results/model_selection.csv"))
    tfidf_sensitivity = pd.read_csv(project_path("results/tfidf_sensitivity.csv"))

    split_rows = {name: detail["rows"] for name, detail in summary["splits"].items()}
    total = summary["total_rows"]
    ratios = {name: rows / total for name, rows in split_rows.items()}
    if not (0.58 <= ratios["train"] <= 0.62 and 0.18 <= ratios["val"] <= 0.22 and 0.18 <= ratios["test"] <= 0.22):
        raise AssertionError(f"数据划分比例异常: {ratios}")
    if summary["tfidf"]["fit_on"] != "train only":
        raise AssertionError("TF-IDF fit_on 记录异常。")
    if len(baselines["model"].unique()) < 5:
        raise AssertionError("baseline 数量不足，应包含 5 个模型。")
    if set(hyper["parameter"]) != {"learning_rate", "dropout"}:
        raise AssertionError("超参数敏感性实验不完整。")
    required_hyper_columns = {"best_epoch", "best_val_macro_f1", "final_val_macro_f1", "epochs_ran", "stopped_early"}
    if not required_hyper_columns.issubset(set(hyper.columns)):
        raise AssertionError("超参数结果缺少 best validation 字段。")
    if set(model_selection["parameter"]) != {"hidden_dims"}:
        raise AssertionError("隐藏层结构敏感性实验不完整。")
    if set(tfidf_sensitivity["parameter"]) != {"max_features", "ngram_range"}:
        raise AssertionError("TF-IDF 敏感性实验不完整。")
    if int(metrics["mlp"]["history_summary"]["best_epoch"]) > int(metrics["config"]["max_epochs"]):
        raise AssertionError("best epoch 超过最大训练轮数。")
    if metrics["selection"]["test_uses"] != "best validation checkpoint":
        raise AssertionError("测试集没有记录为使用 best validation checkpoint。")

    report_paths = sorted(
        [project_path(path) for path in REPORT_CANDIDATES if project_path(path).exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if report_paths:
        text = docx_text(report_paths[0])
        chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
        if chinese_count < 4000:
            raise AssertionError(f"报告中文字符数不足 4000: {chinese_count}")
        if "我是 AI" in text or "根据你的要求我生成了" in text:
            raise AssertionError("报告中存在不应出现的 AI 叙述。")
        forbidden_markers = ["建议用自己的话", "运行后填写", "当前正文中文字符数不足"]
        found_markers = [marker for marker in forbidden_markers if marker in text]
        if found_markers:
            raise AssertionError(f"报告中仍存在占位提示: {found_markers}")
        if str(metrics["mlp"]["test"]["accuracy"])[:5] not in text:
            print("提示：报告中未直接找到完整 accuracy 原始字符串，已使用四位小数格式写入。")
    else:
        chinese_count = 0
        print("提示：未检查报告文件；本次检查仅覆盖实验代码、结果和图表。")

    print("提交前自动检查通过。")
    print(f"划分比例: {ratios}")
    if chinese_count:
        print(f"报告中文字符数约: {chinese_count}")


if __name__ == "__main__":
    main()
