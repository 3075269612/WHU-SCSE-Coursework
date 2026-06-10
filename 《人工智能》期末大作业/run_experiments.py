from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import load_config
from src.data import (
    clean_dataset,
    dataset_summary,
    load_sms_dataset,
    make_exploration_figures,
    save_splits,
    split_dataset,
    vectorize_splits,
)
from src.evaluation import classification_metrics
from src.plots import (
    save_confusion_matrix,
    save_sensitivity_plot,
    save_tfidf_sensitivity_plot,
    save_training_curve,
)
from src.training import predict_mlp, train_mlp
from src.utils import PROJECT_ROOT, ensure_directories, project_path, set_seed, write_json


def save_csv_with_fallback(df: pd.DataFrame, path: str) -> str:
    output = project_path(path)
    try:
        df.to_csv(output, index=False, encoding="utf-8")
        return path
    except PermissionError:
        fallback = output.with_name(f"{output.stem}_optimized_{int(time.time())}{output.suffix}")
        df.to_csv(fallback, index=False, encoding="utf-8")
        print(f"Warning: {output} 正在被占用，已另存为 {fallback}")
        return str(fallback.relative_to(PROJECT_ROOT))


def save_json_with_fallback(data: dict[str, Any], path: str) -> str:
    output = project_path(path)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        return path
    except PermissionError:
        fallback = output.with_name(f"{output.stem}_optimized_{int(time.time())}{output.suffix}")
        fallback.write_text(payload, encoding="utf-8")
        print(f"Warning: {output} 正在被占用，已另存为 {fallback}")
        return str(fallback.relative_to(PROJECT_ROOT))


def class_weights_from_labels(y_train: np.ndarray) -> np.ndarray:
    """按 sklearn balanced 规则计算类别权重，提升少数类 spam 的损失贡献。"""
    classes, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (len(classes) * counts)
    output = np.ones(int(classes.max()) + 1, dtype="float32")
    output[classes.astype(int)] = weights.astype("float32")
    return output


def run_baselines(x_train, y_train, x_val, y_val, x_test, y_test, seed: int) -> pd.DataFrame:
    baselines = [
        ("majority_class", DummyClassifier(strategy="most_frequent")),
        ("random_stratified", DummyClassifier(strategy="stratified", random_state=seed)),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
        ("multinomial_nb", MultinomialNB()),
        ("linear_svm", LinearSVC(random_state=seed, class_weight="balanced", max_iter=5000)),
    ]
    rows = []
    for name, model in baselines:
        model.fit(x_train, y_train)
        for split_name, features, labels in [("val", x_val, y_val), ("test", x_test, y_test)]:
            pred = model.predict(features)
            metrics = classification_metrics(labels, pred)
            rows.append({"model": name, "split": split_name, **metrics})
    return pd.DataFrame(rows)


def summarize_history(history: dict, run_result: dict) -> dict:
    if history["epoch"]:
        best_idx = int(pd.Series(history["val_macro_f1"]).idxmax())
        history_summary = {
            "history_best_epoch": int(history["epoch"][best_idx]),
            "history_best_val_macro_f1": float(history["val_macro_f1"][best_idx]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "final_train_accuracy": float(history["train_accuracy"][-1]),
            "final_val_accuracy": float(history["val_accuracy"][-1]),
            "final_val_macro_f1": float(history["val_macro_f1"][-1]),
        }
    else:
        history_summary = {}
    history_summary.update({
        "best_epoch": int(run_result["best_epoch"]),
        "best_val_loss": float(run_result["best_val_loss"]),
        "best_val_macro_f1": float(run_result["best_val_metrics"]["macro_f1"]),
        "best_val_accuracy": float(run_result["best_val_metrics"]["accuracy"]),
        "epochs_ran": int(run_result["epochs_ran"]),
        "stopped_early": bool(run_result["stopped_early"]),
    })
    return history_summary


def run_single_mlp(
    config: dict,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    hidden_dims: list[int],
    lr: float,
    dropout: float,
    seed: int,
    track_history: bool,
    class_weights: np.ndarray | None,
) -> dict:
    model_config = config["model"]
    return train_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim=x_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=float(dropout),
        learning_rate=float(lr),
        weight_decay=float(model_config["weight_decay"]),
        batch_size=int(model_config["batch_size"]),
        epochs=int(model_config["epochs"]),
        seed=seed,
        device_name="cpu",
        track_history=track_history,
        patience=int(model_config["early_stopping_patience"]),
        class_weights=class_weights if bool(model_config["use_class_weights"]) else None,
    )


def run_candidate(
    *,
    config: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    hidden_dims: list[int],
    lr: float,
    dropout: float,
    max_features: int,
    ngram_range: tuple[int, int],
    track_history: bool,
) -> tuple[dict, Any, Any, Any, Any, Any, Any, Any]:
    vectorizer, x_train, x_val, x_test, y_train, y_val, y_test = vectorize_splits(
        train_df,
        val_df,
        test_df,
        config,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    class_weights = class_weights_from_labels(y_train)
    result = run_single_mlp(
        config,
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dims=hidden_dims,
        lr=lr,
        dropout=dropout,
        seed=seed,
        track_history=track_history,
        class_weights=class_weights,
    )
    return result, vectorizer, x_train, x_val, x_test, y_train, y_val, y_test


def result_row(parameter: str, value: Any, result: dict, controls: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameter": parameter,
        "value": str(value),
        **controls,
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "stopped_early": bool(result["stopped_early"]),
        "best_val_accuracy": result["best_val_metrics"]["accuracy"],
        "best_val_macro_f1": result["best_val_metrics"]["macro_f1"],
        "best_val_macro_precision": result["best_val_metrics"]["macro_precision"],
        "best_val_macro_recall": result["best_val_metrics"]["macro_recall"],
        "final_val_accuracy": result["final_val_metrics"]["accuracy"],
        "final_val_macro_f1": result["final_val_metrics"]["macro_f1"],
        "final_val_macro_precision": result["final_val_metrics"]["macro_precision"],
        "final_val_macro_recall": result["final_val_metrics"]["macro_recall"],
        # 兼容既有报告/绘图字段：这里的 val_* 明确指 best validation 指标。
        "val_accuracy": result["best_val_metrics"]["accuracy"],
        "val_macro_f1": result["best_val_metrics"]["macro_f1"],
        "val_macro_precision": result["best_val_metrics"]["macro_precision"],
        "val_macro_recall": result["best_val_metrics"]["macro_recall"],
    }


def best_value(rows: list[dict[str, Any]], parameter: str) -> str:
    df = pd.DataFrame([row for row in rows if row["parameter"] == parameter])
    best = df.sort_values("best_val_macro_f1", ascending=False).iloc[0]
    return str(best["value"])


def parse_ngram(value: str) -> tuple[int, int]:
    cleaned = value.strip().strip("()[]")
    left, right = cleaned.split(",")
    return int(left), int(right)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    ensure_directories(
        config["outputs"]["results_dir"],
        config["outputs"]["figures_dir"],
        config["outputs"]["report_dir"],
        "data/processed",
    )

    dataset_csv = project_path(config["data"]["dataset_csv"])
    if not dataset_csv.exists():
        raise FileNotFoundError("未找到数据文件，请先运行 scripts/download_data.py")

    raw_df = load_sms_dataset(dataset_csv)
    df, cleaning = clean_dataset(raw_df)
    train_df, val_df, test_df = split_dataset(
        df,
        seed=seed,
        test_size=float(config["data"]["test_size"]),
        val_size_from_train_val=float(config["data"]["val_size_from_train_val"]),
    )
    save_splits(train_df, val_df, test_df, config)
    exploration_figures = make_exploration_figures(df, train_df, config)

    default_hidden_dims = [int(value) for value in config["model"]["hidden_dims"]]
    default_lr = float(config["model"]["learning_rate"])
    default_dropout = float(config["model"]["dropout"])
    default_max_features = int(config["features"]["max_features"])
    default_ngram = (int(config["features"]["ngram_min"]), int(config["features"]["ngram_max"]))

    hyper_rows: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    tfidf_rows: list[dict[str, Any]] = []

    print("Learning-rate sensitivity with best checkpoint...")
    for lr in config["experiments"]["learning_rates"]:
        result, *_ = run_candidate(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            hidden_dims=default_hidden_dims,
            lr=float(lr),
            dropout=default_dropout,
            max_features=default_max_features,
            ngram_range=default_ngram,
            track_history=False,
        )
        hyper_rows.append(result_row("learning_rate", float(lr), result, {
            "controlled_hidden_dims": str(default_hidden_dims),
            "controlled_dropout": default_dropout,
            "controlled_learning_rate": default_lr,
            "controlled_max_features": default_max_features,
            "controlled_ngram_range": str(default_ngram),
        }))
    selected_lr = float(best_value(hyper_rows, "learning_rate"))

    print("Dropout sensitivity with best checkpoint...")
    for dropout in config["experiments"]["dropouts"]:
        result, *_ = run_candidate(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            hidden_dims=default_hidden_dims,
            lr=selected_lr,
            dropout=float(dropout),
            max_features=default_max_features,
            ngram_range=default_ngram,
            track_history=False,
        )
        hyper_rows.append(result_row("dropout", float(dropout), result, {
            "controlled_hidden_dims": str(default_hidden_dims),
            "controlled_dropout": default_dropout,
            "controlled_learning_rate": selected_lr,
            "controlled_max_features": default_max_features,
            "controlled_ngram_range": str(default_ngram),
        }))
    selected_dropout = float(best_value(hyper_rows, "dropout"))

    print("Hidden-dimension sensitivity with best checkpoint...")
    for hidden_dims in config["experiments"]["hidden_dims"]:
        hidden_dims = [int(value) for value in hidden_dims]
        result, *_ = run_candidate(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            hidden_dims=hidden_dims,
            lr=selected_lr,
            dropout=selected_dropout,
            max_features=default_max_features,
            ngram_range=default_ngram,
            track_history=False,
        )
        model_selection_rows.append(result_row("hidden_dims", hidden_dims, result, {
            "controlled_dropout": selected_dropout,
            "controlled_learning_rate": selected_lr,
            "controlled_max_features": default_max_features,
            "controlled_ngram_range": str(default_ngram),
        }))
    selected_hidden = [int(value) for value in ast.literal_eval(best_value(model_selection_rows, "hidden_dims"))]

    print("TF-IDF sensitivity with best checkpoint...")
    for max_features in config["experiments"]["tfidf_max_features"]:
        result, *_ = run_candidate(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            hidden_dims=selected_hidden,
            lr=selected_lr,
            dropout=selected_dropout,
            max_features=int(max_features),
            ngram_range=default_ngram,
            track_history=False,
        )
        tfidf_rows.append(result_row("max_features", int(max_features), result, {
            "controlled_hidden_dims": str(selected_hidden),
            "controlled_dropout": selected_dropout,
            "controlled_learning_rate": selected_lr,
            "controlled_max_features": default_max_features,
            "controlled_ngram_range": str(default_ngram),
        }))
    selected_max_features = int(best_value(tfidf_rows, "max_features"))

    for ngram in config["experiments"]["tfidf_ngram_ranges"]:
        ngram_tuple = (int(ngram[0]), int(ngram[1]))
        result, *_ = run_candidate(
            config=config,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=seed,
            hidden_dims=selected_hidden,
            lr=selected_lr,
            dropout=selected_dropout,
            max_features=selected_max_features,
            ngram_range=ngram_tuple,
            track_history=False,
        )
        tfidf_rows.append(result_row("ngram_range", ngram_tuple, result, {
            "controlled_hidden_dims": str(selected_hidden),
            "controlled_dropout": selected_dropout,
            "controlled_learning_rate": selected_lr,
            "controlled_max_features": selected_max_features,
            "controlled_ngram_range": str(default_ngram),
        }))
    selected_ngram = parse_ngram(best_value(tfidf_rows, "ngram_range"))

    print("Training final selected MLP with full history...")
    final_run, vectorizer, x_train, x_val, x_test, y_train, y_val, y_test = run_candidate(
        config=config,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        seed=seed,
        hidden_dims=selected_hidden,
        lr=selected_lr,
        dropout=selected_dropout,
        max_features=selected_max_features,
        ngram_range=selected_ngram,
        track_history=True,
    )
    y_test_pred = predict_mlp(final_run["model"], x_test, batch_size=256, device_name="cpu")
    test_metrics = classification_metrics(y_test, y_test_pred)

    baseline_df = run_baselines(x_train, y_train, x_val, y_val, x_test, y_test, seed)
    baseline_file = save_csv_with_fallback(baseline_df, "results/baselines.csv")

    history_df = pd.DataFrame(final_run["history"])
    training_history_file = save_csv_with_fallback(history_df, "results/training_history.csv")
    save_training_curve(final_run["history"], "figures/training_curve.png")
    save_confusion_matrix(y_test, y_test_pred, "figures/confusion_matrix.png")

    hyper_df = pd.DataFrame(hyper_rows)
    model_selection_df = pd.DataFrame(model_selection_rows)
    tfidf_df = pd.DataFrame(tfidf_rows)
    hyperparams_file = save_csv_with_fallback(hyper_df, "results/hyperparams.csv")
    model_selection_file = save_csv_with_fallback(model_selection_df, "results/model_selection.csv")
    tfidf_sensitivity_file = save_csv_with_fallback(tfidf_df, "results/tfidf_sensitivity.csv")
    save_sensitivity_plot(hyper_df, "learning_rate", "figures/lr_sensitivity.png", "学习率敏感性分析")
    save_sensitivity_plot(hyper_df, "dropout", "figures/dropout_sensitivity.png", "Dropout 敏感性分析")
    save_sensitivity_plot(model_selection_df, "hidden_dims", "figures/hidden_dims_sensitivity.png", "隐藏层结构敏感性分析")
    save_tfidf_sensitivity_plot(tfidf_df, "figures/tfidf_sensitivity.png")

    summary = dataset_summary(df, train_df, val_df, test_df, cleaning)
    summary["tfidf"] = {
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "max_features": selected_max_features,
        "ngram_range": [selected_ngram[0], selected_ngram[1]],
        "fit_on": "train only",
        "transform_on": ["val", "test"],
    }
    summary["figures"] = exploration_figures
    dataset_summary_file = save_json_with_fallback(summary, "results/dataset_summary.json")

    metrics = {
        "project": "垃圾短信识别",
        "student": {"name": "王李明", "id": "2024302181194"},
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "config": {
            "seed": seed,
            "hidden_dims": selected_hidden,
            "dropout": selected_dropout,
            "learning_rate": selected_lr,
            "batch_size": int(config["model"]["batch_size"]),
            "max_epochs": int(config["model"]["epochs"]),
            "early_stopping_patience": int(config["model"]["early_stopping_patience"]),
            "weight_decay": float(config["model"]["weight_decay"]),
            "use_class_weights": bool(config["model"]["use_class_weights"]),
            "class_weights": final_run["used_class_weights"],
            "tfidf_max_features": selected_max_features,
            "tfidf_ngram_range": [selected_ngram[0], selected_ngram[1]],
        },
        "selection": {
            "selected_learning_rate": selected_lr,
            "selected_dropout": selected_dropout,
            "selected_hidden_dims": selected_hidden,
            "selected_max_features": selected_max_features,
            "selected_ngram_range": [selected_ngram[0], selected_ngram[1]],
            "criterion": "best validation macro_f1",
            "test_uses": "best validation checkpoint",
        },
        "dataset_summary_file": dataset_summary_file,
        "baseline_file": baseline_file,
        "hyperparams_file": hyperparams_file,
        "model_selection_file": model_selection_file,
        "tfidf_sensitivity_file": tfidf_sensitivity_file,
        "training_history_file": training_history_file,
        "figures": {
            "training_curve": "figures/training_curve.png",
            "confusion_matrix": "figures/confusion_matrix.png",
            "lr_sensitivity": "figures/lr_sensitivity.png",
            "dropout_sensitivity": "figures/dropout_sensitivity.png",
            "hidden_dims_sensitivity": "figures/hidden_dims_sensitivity.png",
            "tfidf_sensitivity": "figures/tfidf_sensitivity.png",
            **exploration_figures,
        },
        "mlp": {
            "validation": final_run["best_val_metrics"],
            "final_epoch_validation": final_run["final_val_metrics"],
            "test": test_metrics,
            "history_summary": summarize_history(final_run["history"], final_run),
        },
    }
    save_json_with_fallback(metrics, "results/metrics.json")
    print("实验完成，结果已保存到 results/ 和 figures/。")


if __name__ == "__main__":
    main()
