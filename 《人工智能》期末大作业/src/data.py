from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.utils import project_path


LABEL_TO_ID = {"ham": 0, "spam": 1}
ID_TO_LABEL = {0: "ham", 1: "spam"}


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _safe_savefig(path: Path, *, dpi: int) -> None:
    try:
        plt.savefig(path, dpi=dpi)
    except PermissionError:
        print(f"Warning: {path} 正在被占用，保留旧图。")


def load_sms_dataset(csv_path: str | Path) -> pd.DataFrame:
    """加载整理后的短信数据，并检查标签合法性。"""
    df = pd.read_csv(project_path(csv_path))
    expected_columns = {"label", "text"}
    if set(df.columns) != expected_columns:
        raise ValueError(f"数据列应为 {expected_columns}, 实际为 {set(df.columns)}")
    invalid_labels = sorted(set(df["label"]) - set(LABEL_TO_ID))
    if invalid_labels:
        raise ValueError(f"发现未知标签: {invalid_labels}")
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """执行缺失值检测和 dropna，保留处理记录用于报告。"""
    missing_before = int(df.isna().sum().sum())
    rows_before = int(len(df))
    clean_df = df.dropna().drop_duplicates().reset_index(drop=True)
    missing_after = int(clean_df.isna().sum().sum())
    rows_after = int(len(clean_df))
    return clean_df, {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_removed": rows_before - rows_after,
    }


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size_from_train_val: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """先分层划分数据集，再进行后续特征提取，避免数据泄漏。"""
    y = df["label"]
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_from_train_val,
        stratify=train_val_df["label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, config: dict) -> None:
    data_config = config["data"]
    for frame, path_key in [
        (train_df, "train_csv"),
        (val_df, "val_csv"),
        (test_df, "test_csv"),
    ]:
        output_path = project_path(data_config[path_key])
        try:
            frame.to_csv(output_path, index=False, encoding="utf-8")
        except PermissionError:
            print(f"Warning: {output_path} 正在被占用，跳过覆盖。")


def build_vectorizer(
    config: dict,
    *,
    max_features: int | None = None,
    ngram_range: tuple[int, int] | None = None,
) -> TfidfVectorizer:
    features = config["features"]
    if max_features is None:
        max_features = int(features["max_features"])
    if ngram_range is None:
        ngram_range = (int(features["ngram_min"]), int(features["ngram_max"]))
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=int(features["min_df"]),
        sublinear_tf=bool(features["sublinear_tf"]),
        lowercase=True,
        strip_accents="unicode",
    )


def vectorize_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
    *,
    max_features: int | None = None,
    ngram_range: tuple[int, int] | None = None,
):
    """TF-IDF 只在训练集 fit；验证集和测试集只 transform。"""
    vectorizer = build_vectorizer(config, max_features=max_features, ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_val = vectorizer.transform(val_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"].map(LABEL_TO_ID).to_numpy()
    y_val = val_df["label"].map(LABEL_TO_ID).to_numpy()
    y_test = test_df["label"].map(LABEL_TO_ID).to_numpy()
    return vectorizer, x_train, x_val, x_test, y_train, y_val, y_test


def dataset_summary(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cleaning: dict) -> dict:
    def label_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {key: int(value) for key, value in frame["label"].value_counts().sort_index().items()}

    return {
        "total_rows": int(len(df)),
        "class_counts": label_counts(df),
        "cleaning": cleaning,
        "splits": {
            "train": {"rows": int(len(train_df)), "class_counts": label_counts(train_df)},
            "val": {"rows": int(len(val_df)), "class_counts": label_counts(val_df)},
            "test": {"rows": int(len(test_df)), "class_counts": label_counts(test_df)},
        },
        "text_length": {
            "mean": float(df["text"].str.len().mean()),
            "median": float(df["text"].str.len().median()),
            "max": int(df["text"].str.len().max()),
            "min": int(df["text"].str.len().min()),
        },
    }


def make_exploration_figures(df: pd.DataFrame, train_df: pd.DataFrame, config: dict) -> dict[str, str]:
    figures_dir = project_path(config["outputs"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    label_counts = df["label"].value_counts().reindex(["ham", "spam"])
    plt.figure(figsize=(6, 4))
    label_counts.plot(kind="bar", color=["#4C78A8", "#E45756"])
    plt.title("短信类别分布")
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = figures_dir / "label_distribution.png"
    _safe_savefig(path, dpi=160)
    plt.close()
    paths["label_distribution"] = str(path.relative_to(project_path(".")))

    lengths = df.assign(length=df["text"].str.len())
    plt.figure(figsize=(7, 4))
    for label, color in [("ham", "#4C78A8"), ("spam", "#E45756")]:
        plt.hist(lengths.loc[lengths["label"] == label, "length"], bins=40, alpha=0.65, label=label, color=color)
    plt.title("短信长度分布")
    plt.xlabel("字符数")
    plt.ylabel("频数")
    plt.legend()
    plt.tight_layout()
    path = figures_dir / "message_length_distribution.png"
    _safe_savefig(path, dpi=160)
    plt.close()
    paths["message_length_distribution"] = str(path.relative_to(project_path(".")))

    # 仅用训练集统计词频，避免把验证/测试文本内容泄漏进特征解释。
    count_vectorizer = CountVectorizer(stop_words="english", max_features=15, lowercase=True)
    counts = count_vectorizer.fit_transform(train_df["text"])
    words = count_vectorizer.get_feature_names_out()
    totals = counts.sum(axis=0).A1
    top_terms = pd.Series(totals, index=words).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    top_terms.plot(kind="barh", color="#59A14F")
    plt.title("训练集高频词分布")
    plt.xlabel("出现次数")
    plt.ylabel("词项")
    plt.tight_layout()
    path = figures_dir / "top_terms.png"
    _safe_savefig(path, dpi=160)
    plt.close()
    paths["top_terms"] = str(path.relative_to(project_path(".")))
    return paths
