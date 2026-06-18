from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"


def load_histories() -> dict[str, list[dict[str, float]]]:
    return json.loads(
        (RESULTS_DIR / "training_history.json").read_text(encoding="utf-8")
    )


def load_summaries() -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    with (RESULTS_DIR / "parameter_comparison.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            summaries[row["name"]] = {
                key: float(value)
                for key, value in row.items()
                if key
                in {
                    "parameters",
                    "elapsed_seconds",
                    "first_perfect_epoch",
                    "loss",
                    "mlm_loss",
                    "nsp_loss",
                    "mlm_accuracy",
                    "nsp_accuracy",
                }
            }
    return summaries


def save_training_curves(histories: dict[str, list[dict[str, float]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["loss"] for row in history], label=name)
        axes[1].plot(
            epochs, [row["mlm_accuracy"] for row in history], label=f"{name} MLM"
        )
        axes[1].plot(
            epochs,
            [row["nsp_accuracy"] for row in history],
            linestyle="--",
            label=f"{name} NSP",
        )
    axes[0].set(title="Pretraining loss", xlabel="Epoch", ylabel="Loss")
    axes[1].set(
        title="Task accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        ylim=(-0.05, 1.05),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "training_curves.png", dpi=180)
    plt.close(figure)


def save_parameter_comparison(summaries: dict[str, dict[str, float]]) -> None:
    names = list(summaries)
    convergence_epochs = [summaries[name]["first_perfect_epoch"] for name in names]
    parameters = [summaries[name]["parameters"] / 1_000_000 for name in names]
    colors = ["#2E74B5", "#70AD47", "#ED7D31"][: len(names)]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, convergence_epochs, color=colors)
    axes[0].set(
        title="First epoch with perfect MLM and NSP accuracy",
        ylabel="Epoch",
    )
    axes[1].bar(names, parameters, color=colors)
    axes[1].set(title="Model size", ylabel="Parameters (million)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "parameter_comparison.png", dpi=180)
    plt.close(figure)


def save_architecture_figure() -> None:
    figure, axis = plt.subplots(figsize=(11, 2.8))
    axis.axis("off")
    labels = [
        "Sentence pair",
        "Token + Position\n+ Segment embedding",
        "Transformer\nencoder stack",
        "[MASK] states\nMLM head",
        "[CLS] state\nNSP head",
    ]
    x_positions = [0.04, 0.25, 0.49, 0.74, 0.74]
    y_positions = [0.5, 0.5, 0.5, 0.7, 0.25]
    widths = [0.15, 0.18, 0.17, 0.16, 0.16]
    colors = ["#E8EEF5", "#D9EAD3", "#FFF2CC", "#DDEBF7", "#FCE4D6"]
    for x, y, width, label, color in zip(
        x_positions, y_positions, widths, labels, colors
    ):
        axis.add_patch(
            plt.Rectangle(
                (x, y - 0.12),
                width,
                0.24,
                facecolor=color,
                edgecolor="#4A4A4A",
                linewidth=1.2,
            )
        )
        axis.text(x + width / 2, y, label, ha="center", va="center", fontsize=10)
    arrow_pairs = [
        ((0.19, 0.5), (0.25, 0.5)),
        ((0.43, 0.5), (0.49, 0.5)),
        ((0.66, 0.5), (0.74, 0.7)),
        ((0.66, 0.5), (0.74, 0.25)),
    ]
    for start, end in arrow_pairs:
        axis.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4}
        )
    axis.set_xlim(0, 0.95)
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "bert_pretraining_pipeline.png", dpi=180)
    plt.close(figure)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_training_curves(load_histories())
    save_parameter_comparison(load_summaries())
    save_architecture_figure()
    print(f"Figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
