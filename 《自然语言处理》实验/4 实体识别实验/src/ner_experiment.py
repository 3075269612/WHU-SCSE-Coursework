from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT_DIR / "docs" / "reference" / "4实体识别实验代码"
DATA_DIR = ROOT_DIR / "data" / "raw" / "ResumeNER"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

sys.path.insert(0, str(REFERENCE_DIR))

from data import build_corpus  # noqa: E402
from evaluating import Metrics  # noqa: E402
from models.config import TrainingConfig  # noqa: E402
from utils import extend_maps, load_model, prepocess_data_for_crf  # noqa: E402


MODEL_SPECS = [
    ("lstm", "BiLSTM", False),
    ("bilstm+CRF+w2v", "BiLSTM+CRF+w2v", True),
    ("cnn", "CNN", False),
    ("cnn+CRF", "CNN+CRF", True),
    ("cnn+CRF+w2v", "CNN+CRF+w2v", True),
]


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_test_data():
    TrainingConfig.pretrained_emb = str(DATA_DIR / "pretrained_word_emb" / "word2vec.txt")
    train_words, train_tags, word2id, tag2id, word_emb = build_corpus(
        "train", data_dir=str(DATA_DIR)
    )
    dev_words, dev_tags = build_corpus("dev", make_vocab=False, data_dir=str(DATA_DIR))
    test_words, test_tags = build_corpus("test", make_vocab=False, data_dir=str(DATA_DIR))
    return {
        "train_words": train_words,
        "train_tags": train_tags,
        "dev_words": dev_words,
        "dev_tags": dev_tags,
        "test_words": test_words,
        "test_tags": test_tags,
        "word2id": word2id,
        "tag2id": tag2id,
        "word_emb": word_emb,
    }


def dataset_stats(data: dict) -> dict:
    stats = {}
    for split in ("train", "dev", "test"):
        words = data[f"{split}_words"]
        tags = data[f"{split}_tags"]
        stats[split] = {
            "sentences": len(words),
            "tokens": sum(len(x) for x in words),
            "avg_length": round(sum(len(x) for x in words) / len(words), 2),
            "labels": len({tag for sent in tags for tag in sent}),
        }
    stats["vocab_size"] = len(data["word2id"])
    stats["pretrained_embedding_dim"] = int(data["word_emb"].shape[1])
    return stats


def evaluate_checkpoint(name: str, display_name: str, uses_crf: bool, data: dict) -> dict:
    model_path = MODEL_DIR / f"{name}.pkl"
    model = load_model(str(model_path))
    word2id = data["word2id"]
    tag2id = extend_maps(deepcopy(data["tag2id"]), for_crf=uses_crf)
    test_words = deepcopy(data["test_words"])
    test_tags = deepcopy(data["test_tags"])
    if uses_crf:
        test_words, test_tags = prepocess_data_for_crf(test_words, test_tags, test=True)

    model.model.to(model.device)
    model.best_model.to(model.device)
    if hasattr(model.best_model, "bilstm"):
        model.best_model.bilstm.flatten_parameters()

    with torch.no_grad():
        pred_tags, gold_tags = model.test(test_words, test_tags, word2id, tag2id)

    metrics = Metrics(gold_tags, pred_tags, remove_O=False)
    label_avg = metrics._cal_weighted_average()
    all_ent_p = metrics.correct_tags_number.get("all_ent", 0) / metrics.predict_ent_counter["all_ent"]
    all_ent_r = metrics.correct_tags_number.get("all_ent", 0) / metrics.golden_ent_counter["all_ent"]
    all_ent_f1 = 2 * all_ent_p * all_ent_r / (all_ent_p + all_ent_r + 1e-10)
    result = {
        "checkpoint": model_path.name,
        "model": display_name,
        "label_precision": label_avg["precision"],
        "label_recall": label_avg["recall"],
        "label_f1": label_avg["f1_score"],
        "entity_precision": all_ent_p,
        "entity_recall": all_ent_r,
        "entity_f1": all_ent_f1,
        "entity_support": metrics.golden_ent_counter["all_ent"],
    }
    for ent in sorted(metrics.ent_tags):
        result[f"{ent}_f1"] = metrics.f1_scores[ent]
    return result


def write_outputs(stats: dict, rows: list[dict]) -> None:
    with (RESULTS_DIR / "dataset_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys())
    with (RESULTS_DIR / "ner_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["中文简历命名实体识别实验评估结果", "", "数据集统计："]
    for split, item in stats.items():
        if isinstance(item, dict):
            lines.append(
                f"- {split}: {item['sentences']} 句，{item['tokens']} 字，平均长度 {item['avg_length']}，标签数 {item['labels']}"
            )
    lines.append(f"- 词表规模：{stats['vocab_size']}")
    lines.append(f"- 预训练词向量维度：{stats['pretrained_embedding_dim']}")
    lines.append("")
    lines.append("模型对比：")
    for row in rows:
        lines.append(
            f"- {row['model']}: 实体级 P={row['entity_precision']:.4f}, "
            f"R={row['entity_recall']:.4f}, F1={row['entity_f1']:.4f}; "
            f"标签级 F1={row['label_f1']:.4f}"
        )
    best = max(rows, key=lambda x: x["entity_f1"])
    lines.append("")
    lines.append(f"实体级 F1 最高模型：{best['model']}（{best['entity_f1']:.4f}）。")
    (RESULTS_DIR / "experiment_results.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(rows: list[dict]) -> None:
    names = [row["model"] for row in rows]
    entity_f1 = [row["entity_f1"] for row in rows]
    label_f1 = [row["label_f1"] for row in rows]
    x = range(len(rows))

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([i - 0.18 for i in x], entity_f1, width=0.36, label="实体级F1", color="#2f6f73")
    ax.bar([i + 0.18 for i in x], label_f1, width=0.36, label="标签级F1", color="#b7791f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("NER模型评估指标对比")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ner_model_f1_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResumeNER checkpoints.")
    parser.add_argument("--skip-plots", action="store_true", help="Only write text and CSV results.")
    args = parser.parse_args()
    ensure_dirs()
    data = load_test_data()
    stats = dataset_stats(data)
    rows = [evaluate_checkpoint(*spec, data) for spec in MODEL_SPECS]
    write_outputs(stats, rows)
    if not args.skip_plots:
        plot_metrics(rows)
    print((RESULTS_DIR / "experiment_results.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
