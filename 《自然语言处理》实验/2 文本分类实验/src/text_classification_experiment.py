from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw" / "THUCNews"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

PAD = "<PAD>"
UNK = "<UNK>"
MAX_VOCAB_SIZE = 10000


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    sweep_name: str
    sweep_value: str
    batch_size: int = 32
    embed_dim: int = 128
    hidden_size: int = 128
    learning_rate: float = 1e-3
    dropout: float = 0.3
    epochs: int = 4
    pad_size: int = 32
    seed: int = 42
    num_filters: int = 128


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv2d(1, config.hidden_size, (kernel_size, config.embed_dim))
            for kernel_size in (2, 3, 4)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            feature = F.relu(conv(embedded)).squeeze(3)
            pooled.append(F.max_pool1d(feature, feature.size(2)).squeeze(2))
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        combined = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(combined))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run THUCNews TextCNN/BiLSTM classification sweeps.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-train-size", type=int, default=0, help="0 means full training set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_labeled_file(path: Path) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            texts.append(text)
            labels.append(int(label))
    return texts, labels


def limit_balanced(texts: list[str], labels: list[int], max_size: int, seed: int) -> tuple[list[str], list[int]]:
    if max_size <= 0 or max_size >= len(texts):
        return texts, labels
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    per_label = max(1, max_size // len(by_label))
    selected: list[int] = []
    for label, indices in sorted(by_label.items()):
        sample = indices[:]
        rng.shuffle(sample)
        selected.extend(sample[:per_label])
    selected.sort()
    return [texts[i] for i in selected], [labels[i] for i in selected]


def build_vocab(texts: Iterable[str], max_size: int = MAX_VOCAB_SIZE) -> dict[str, int]:
    counts: dict[str, int] = {}
    for text in texts:
        for char in text:
            counts[char] = counts.get(char, 0) + 1
    vocab_items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max_size - 2]
    vocab = {PAD: 0, UNK: 1}
    vocab.update({char: idx + 2 for idx, (char, _) in enumerate(vocab_items)})
    return vocab


def encode_texts(texts: list[str], labels: list[int], vocab: dict[str, int], pad_size: int) -> TensorDataset:
    encoded: list[list[int]] = []
    for text in texts:
        ids = [vocab.get(char, vocab[UNK]) for char in text[:pad_size]]
        if len(ids) < pad_size:
            ids.extend([vocab[PAD]] * (pad_size - len(ids)))
        encoded.append(ids)
    return TensorDataset(torch.tensor(encoded, dtype=torch.long), torch.tensor(labels, dtype=torch.long))


def make_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def build_model(config: ExperimentConfig, vocab_size: int, num_classes: int) -> nn.Module:
    if config.model_name == "TextCNN":
        return TextCNN(vocab_size, num_classes, config)
    if config.model_name == "BiLSTM":
        return BiLSTM(vocab_size, num_classes, config)
    raise ValueError(f"Unknown model: {config.model_name}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            losses.append(loss.item() * batch_y.size(0))
            all_true.extend(batch_y.cpu().numpy().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    total = len(all_true)
    return sum(losses) / total, accuracy_score(all_true, all_pred), np.array(all_true), np.array(all_pred)


def train_one(
    config: ExperimentConfig,
    train_dataset: TensorDataset,
    dev_dataset: TensorDataset,
    test_dataset: TensorDataset,
    vocab_size: int,
    class_names: list[str],
    device: torch.device,
) -> dict:
    set_seed(config.seed)
    model = build_model(config, vocab_size, len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = make_loader(train_dataset, config.batch_size, True, config.seed)
    dev_loader = make_loader(dev_dataset, config.batch_size, False, config.seed)
    test_loader = make_loader(test_dataset, config.batch_size, False, config.seed)
    history: list[dict] = []
    best_state = copy.deepcopy(model.state_dict())
    best_dev_acc = -1.0
    best_dev_loss = float("inf")
    start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_true: list[int] = []
        train_pred: list[int] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_y.size(0)
            train_true.extend(batch_y.cpu().numpy().tolist())
            train_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

        train_loss = train_loss_sum / len(train_true)
        train_acc = accuracy_score(train_true, train_pred)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )
        if dev_acc > best_dev_acc or (dev_acc == best_dev_acc and dev_loss < best_dev_loss):
            best_dev_acc = dev_acc
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    elapsed = time.time() - start
    return {
        "config": config,
        "history": history,
        "best_dev_acc": best_dev_acc,
        "best_dev_loss": best_dev_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_seconds": elapsed,
        "state_dict": best_state,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def sweep_configs(base_epochs: int, seed: int) -> list[ExperimentConfig]:
    base = ExperimentConfig(model_name="", sweep_name="baseline", sweep_value="baseline", epochs=base_epochs, seed=seed)
    sweeps: dict[str, list] = {
        "batch_size": [8, 16, 32, 64],
        "embedding_size": [64, 128, 256],
        "hidden_size": [64, 128, 256],
        "learning_rate": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    configs: list[ExperimentConfig] = []
    for model_name in ("TextCNN", "BiLSTM"):
        for sweep_name, values in sweeps.items():
            for value in values:
                updates = {
                    "model_name": model_name,
                    "sweep_name": sweep_name,
                    "sweep_value": str(value),
                }
                if sweep_name == "batch_size":
                    updates["batch_size"] = int(value)
                elif sweep_name == "embedding_size":
                    updates["embed_dim"] = int(value)
                elif sweep_name == "hidden_size":
                    updates["hidden_size"] = int(value)
                elif sweep_name == "learning_rate":
                    updates["learning_rate"] = float(value)
                elif sweep_name == "dropout":
                    updates["dropout"] = float(value)
                configs.append(replace(base, **updates))
    return configs


def write_csv(results: list[dict]) -> None:
    history_path = RESULTS_DIR / "training_history.csv"
    metrics_path = RESULTS_DIR / "metrics.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "sweep",
                "value",
                "epoch",
                "batch_size",
                "embed_dim",
                "hidden_size",
                "learning_rate",
                "dropout",
                "train_loss",
                "train_acc",
                "dev_loss",
                "dev_acc",
            ]
        )
        for result in results:
            config = result["config"]
            for row in result["history"]:
                writer.writerow(
                    [
                        config.model_name,
                        config.sweep_name,
                        config.sweep_value,
                        row["epoch"],
                        config.batch_size,
                        config.embed_dim,
                        config.hidden_size,
                        config.learning_rate,
                        config.dropout,
                        f"{row['train_loss']:.6f}",
                        f"{row['train_acc']:.6f}",
                        f"{row['dev_loss']:.6f}",
                        f"{row['dev_acc']:.6f}",
                    ]
                )

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "sweep",
                "value",
                "batch_size",
                "embed_dim",
                "hidden_size",
                "learning_rate",
                "dropout",
                "best_dev_loss",
                "best_dev_acc",
                "test_loss",
                "test_acc",
                "elapsed_seconds",
            ]
        )
        for result in results:
            config = result["config"]
            writer.writerow(
                [
                    config.model_name,
                    config.sweep_name,
                    config.sweep_value,
                    config.batch_size,
                    config.embed_dim,
                    config.hidden_size,
                    config.learning_rate,
                    config.dropout,
                    f"{result['best_dev_loss']:.6f}",
                    f"{result['best_dev_acc']:.6f}",
                    f"{result['test_loss']:.6f}",
                    f"{result['test_acc']:.6f}",
                    f"{result['elapsed_seconds']:.2f}",
                ]
            )


def plot_sweep(results: list[dict], sweep_name: str) -> None:
    subset = [result for result in results if result["config"].sweep_name == sweep_name]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    titles = {
        ("TextCNN", "train_loss"): "TextCNN train loss",
        ("TextCNN", "dev_loss"): "TextCNN validation loss",
        ("BiLSTM", "train_loss"): "BiLSTM train loss",
        ("BiLSTM", "dev_loss"): "BiLSTM validation loss",
    }
    for row_idx, model_name in enumerate(("TextCNN", "BiLSTM")):
        for col_idx, metric in enumerate(("train_loss", "dev_loss")):
            ax = axes[row_idx][col_idx]
            for result in subset:
                config = result["config"]
                if config.model_name != model_name:
                    continue
                epochs = [item["epoch"] for item in result["history"]]
                values = [item[metric] for item in result["history"]]
                ax.plot(epochs, values, marker="o", linewidth=1.6, label=config.sweep_value)
            ax.set_title(titles[(model_name, metric)])
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle(f"Loss curves for {sweep_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / f"loss_{sweep_name}.png", dpi=180)
    plt.close(fig)


def write_summary(
    results: list[dict],
    class_names: list[str],
    dataset_stats: dict,
    best: dict,
    device: torch.device,
) -> None:
    lines: list[str] = []
    lines.append("文本分类实验结果汇总")
    lines.append("=" * 40)
    lines.append(f"运行设备: {device}")
    lines.append(f"类别: {', '.join(class_names)}")
    lines.append(
        "数据规模: train={train}, dev={dev}, test={test}, vocab={vocab}".format(**dataset_stats)
    )
    lines.append("")
    lines.append("全部参数扫描结果:")
    lines.append("model\tsweep\tvalue\tbatch\tembed\thidden\tlr\tdropout\tdev_acc\ttest_acc")
    for result in sorted(results, key=lambda item: (item["config"].model_name, item["config"].sweep_name, -item["test_acc"])):
        config = result["config"]
        lines.append(
            f"{config.model_name}\t{config.sweep_name}\t{config.sweep_value}\t"
            f"{config.batch_size}\t{config.embed_dim}\t{config.hidden_size}\t"
            f"{config.learning_rate:g}\t{config.dropout:g}\t"
            f"{result['best_dev_acc']:.4f}\t{result['test_acc']:.4f}"
        )
    best_config = best["config"]
    lines.append("")
    lines.append("最佳参数组合:")
    lines.append(
        f"模型={best_config.model_name}, 扫描项={best_config.sweep_name}, 取值={best_config.sweep_value}, "
        f"batch_size={best_config.batch_size}, embedding_size={best_config.embed_dim}, "
        f"hidden_size={best_config.hidden_size}, learning_rate={best_config.learning_rate:g}, "
        f"dropout={best_config.dropout:g}"
    )
    lines.append(f"验证集准确率={best['best_dev_acc']:.4f}, 测试集准确率={best['test_acc']:.4f}, 测试损失={best['test_loss']:.4f}")
    lines.append("")
    lines.append("最佳模型分类报告:")
    lines.append(best["classification_report"])
    lines.append("最佳模型混淆矩阵:")
    lines.append(json.dumps(best["confusion_matrix"], ensure_ascii=False))
    (RESULTS_DIR / "experiment_results.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    for directory in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = (args.data_dir / "class.txt").read_text(encoding="utf-8").strip().splitlines()
    train_texts, train_labels = read_labeled_file(args.data_dir / "train.txt")
    dev_texts, dev_labels = read_labeled_file(args.data_dir / "dev.txt")
    test_texts, test_labels = read_labeled_file(args.data_dir / "test.txt")
    train_texts, train_labels = limit_balanced(train_texts, train_labels, args.max_train_size, args.seed)

    vocab = build_vocab(train_texts)
    (PROCESSED_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    base_pad_size = 32
    train_dataset = encode_texts(train_texts, train_labels, vocab, base_pad_size)
    dev_dataset = encode_texts(dev_texts, dev_labels, vocab, base_pad_size)
    test_dataset = encode_texts(test_texts, test_labels, vocab, base_pad_size)

    results: list[dict] = []
    for index, config in enumerate(sweep_configs(args.epochs, args.seed), start=1):
        print(
            f"[{index:02d}] {config.model_name} {config.sweep_name}={config.sweep_value} "
            f"batch={config.batch_size} embed={config.embed_dim} hidden={config.hidden_size} "
            f"lr={config.learning_rate:g} dropout={config.dropout:g}"
        )
        result = train_one(config, train_dataset, dev_dataset, test_dataset, len(vocab), class_names, device)
        print(
            f"     dev_acc={result['best_dev_acc']:.4f} test_acc={result['test_acc']:.4f} "
            f"time={result['elapsed_seconds']:.1f}s"
        )
        results.append(result)

    write_csv(results)
    for sweep_name in ("batch_size", "embedding_size", "hidden_size", "learning_rate", "dropout"):
        plot_sweep(results, sweep_name)

    best = max(
        results,
        key=lambda item: (
            item["best_dev_acc"],
            -item["best_dev_loss"],
        ),
    )
    torch.save(best["state_dict"], MODELS_DIR / "best_text_classifier.pt")
    write_summary(
        results,
        class_names,
        {
            "train": len(train_texts),
            "dev": len(dev_texts),
            "test": len(test_texts),
            "vocab": len(vocab),
        },
        best,
        device,
    )
    print("Best:", best["config"])
    print(RESULTS_DIR / "experiment_results.txt")


if __name__ == "__main__":
    main()
