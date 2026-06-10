from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import ID_TO_LABEL
from src.utils import project_path


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_figure(fig, output: Path, *, dpi: int = 180) -> None:
    try:
        fig.savefig(output, dpi=dpi)
    except PermissionError:
        print(f"Warning: {output} 正在被占用，保留旧图。")


def save_training_curve(history: dict, output_path: str | Path) -> None:
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss 曲线")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy 曲线")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    _save_figure(fig, output, dpi=180)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title("测试集混淆矩阵")
    display.figure_.tight_layout()
    _save_figure(display.figure_, output, dpi=180)
    plt.close(display.figure_)


def save_sensitivity_plot(df: pd.DataFrame, parameter: str, output_path: str | Path, title: str) -> None:
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = df[df["parameter"] == parameter].copy()
    subset["value_label"] = subset["value"].astype(str)
    accuracy_col = "best_val_accuracy" if "best_val_accuracy" in subset.columns else "val_accuracy"
    macro_f1_col = "best_val_macro_f1" if "best_val_macro_f1" in subset.columns else "val_macro_f1"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(subset["value_label"], subset[accuracy_col], marker="o", label="Best Val Accuracy")
    ax.plot(subset["value_label"], subset[macro_f1_col], marker="s", label="Best Val Macro-F1")
    best_idx = subset[macro_f1_col].idxmax()
    best_row = subset.loc[best_idx]
    ax.scatter([str(best_row["value"])], [best_row[macro_f1_col]], color="#E45756", zorder=5, label="Best Macro-F1")
    ax.set_title(title)
    ax.set_xlabel(parameter)
    ax.set_ylabel("Score")
    values = pd.concat([subset[accuracy_col], subset[macro_f1_col]])
    lower = max(0.0, float(values.min()) - 0.02)
    upper = min(1.0, float(values.max()) + 0.02)
    if upper - lower < 0.04:
        lower = max(0.0, upper - 0.04)
    ax.set_ylim(lower, upper)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output, dpi=180)
    plt.close(fig)


def save_tfidf_sensitivity_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, parameter, title in [
        (axes[0], "max_features", "TF-IDF max_features 敏感性"),
        (axes[1], "ngram_range", "TF-IDF ngram_range 敏感性"),
    ]:
        subset = df[df["parameter"] == parameter].copy()
        subset["value_label"] = subset["value"].astype(str)
        ax.plot(subset["value_label"], subset["best_val_accuracy"], marker="o", label="Best Val Accuracy")
        ax.plot(subset["value_label"], subset["best_val_macro_f1"], marker="s", label="Best Val Macro-F1")
        best_idx = subset["best_val_macro_f1"].idxmax()
        best_row = subset.loc[best_idx]
        ax.scatter([str(best_row["value"])], [best_row["best_val_macro_f1"]], color="#E45756", zorder=5, label="Best")
        values = pd.concat([subset["best_val_accuracy"], subset["best_val_macro_f1"]])
        lower = max(0.0, float(values.min()) - 0.02)
        upper = min(1.0, float(values.max()) + 0.02)
        if upper - lower < 0.04:
            lower = max(0.0, upper - 0.04)
        ax.set_ylim(lower, upper)
        ax.set_title(title)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Score")
        ax.legend()
    fig.tight_layout()
    _save_figure(fig, output, dpi=180)
    plt.close(fig)
