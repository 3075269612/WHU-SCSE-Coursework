from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import jieba
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import PCA


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "raw" / "词向量实验数据集.txt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_SEGMENTED_PATH = ROOT_DIR / "data" / "processed" / "segmented_corpus.txt"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "models" / "word2vec.model"
DEFAULT_VECTORS_PATH = DEFAULT_OUTPUT_DIR / "vectors" / "word_vectors.txt"
DEFAULT_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "results" / "experiment_results.txt"
DEFAULT_FIGURE_PATH = DEFAULT_OUTPUT_DIR / "figures" / "pca_word_vectors.png"

SIMILARITY_PAIRS = [
    ("中国", "中华"),
    ("中国", "人民"),
]

MOST_SIMILAR_WORDS = [
    "武汉",
    "中国",
]

ANALOGY_TASKS = [
    (["湖北", "成都"], ["武汉"], "湖北 - 武汉 + 成都"),
    (["江苏", "广州"], ["南京"], "江苏 - 南京 + 广州"),
]

PCA_WORDS = [
    "江苏",
    "南京",
    "成都",
    "四川",
    "湖北",
    "武汉",
    "河南",
    "郑州",
    "甘肃",
    "兰州",
    "湖南",
    "长沙",
    "陕西",
    "西安",
    "吉林",
    "长春",
    "广东",
    "广州",
    "浙江",
    "杭州",
]

PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and analyze Chinese Word2Vec vectors.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Input corpus path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--reuse-segmented",
        action="store_true",
        help="Reuse data/processed/segmented_corpus.txt if it already exists.",
    )
    return parser.parse_args()


def is_valid_token(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    if PUNCTUATION_RE.match(token):
        return False
    return True


def segment_line(line: str) -> list[str]:
    return [token.strip() for token in jieba.cut(line) if is_valid_token(token)]


def segment_corpus(dataset_path: Path, segmented_path: Path) -> tuple[int, int]:
    line_count = 0
    token_count = 0
    with dataset_path.open("r", encoding="utf-8") as src, segmented_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            tokens = segment_line(line)
            if tokens:
                dst.write(" ".join(tokens) + "\n")
                token_count += len(tokens)
            line_count += 1
            if line_count % 100000 == 0:
                print(f"Segmented {line_count} lines...")
    return line_count, token_count


def train_model(segmented_path: Path) -> Word2Vec:
    sentences = LineSentence(str(segmented_path))
    return Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=5,
        seed=42,
    )


def format_most_similar(items: Iterable[tuple[str, float]]) -> str:
    return ", ".join(f"{word}({score:.4f})" for word, score in items)


def add_similarity_results(model: Word2Vec, lines: list[str]) -> None:
    lines.append("1. 词语相似度")
    for left, right in SIMILARITY_PAIRS:
        if left in model.wv and right in model.wv:
            score = model.wv.similarity(left, right)
            lines.append(f"   {left} - {right}: {score:.4f}")
        else:
            missing = [word for word in (left, right) if word not in model.wv]
            lines.append(f"   {left} - {right}: 无法计算，缺失词 {missing}")
    lines.append("")


def add_nearest_results(model: Word2Vec, lines: list[str]) -> None:
    lines.append("2. 最相似词 Top 5")
    for word in MOST_SIMILAR_WORDS:
        if word in model.wv:
            results = model.wv.most_similar(positive=[word], topn=5)
            lines.append(f"   {word}: {format_most_similar(results)}")
        else:
            lines.append(f"   {word}: 无法计算，词不在词表中")
    lines.append("")


def add_analogy_results(model: Word2Vec, lines: list[str]) -> None:
    lines.append("3. 词向量类比 Top 5")
    for positive, negative, label in ANALOGY_TASKS:
        words = positive + negative
        missing = [word for word in words if word not in model.wv]
        if missing:
            lines.append(f"   {label}: 无法计算，缺失词 {missing}")
            continue
        results = model.wv.most_similar(positive=positive, negative=negative, topn=5)
        lines.append(f"   {label}: {format_most_similar(results)}")
    lines.append("")


def run_pca(model: Word2Vec, figure_path: Path, lines: list[str]) -> None:
    available_words = [word for word in PCA_WORDS if word in model.wv]
    missing_words = [word for word in PCA_WORDS if word not in model.wv]
    lines.append("4. PCA 降维与可视化")
    lines.append(f"   参与可视化词数: {len(available_words)}")
    if missing_words:
        lines.append(f"   缺失词: {', '.join(missing_words)}")

    if len(available_words) < 2:
        lines.append("   可用词不足，无法进行 PCA。")
        lines.append("")
        return

    vectors = np.array([model.wv[word] for word in available_words])
    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)
    ratio = pca.explained_variance_ratio_
    lines.append(f"   PCA 方差贡献率: PC1={ratio[0]:.4f}, PC2={ratio[1]:.4f}")
    lines.append("")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="SimHei")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    sns.scatterplot(x=points[:, 0], y=points[:, 1], ax=ax, s=70, color="#2f6fbb")
    for word, point in zip(available_words, points):
        ax.text(point[0], point[1], word, fontsize=10, ha="left", va="bottom")

    ax.set_title("词向量 PCA 二维可视化", fontsize=14)
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path)
    plt.close(fig)


def write_results(
    model: Word2Vec,
    results_path: Path,
    figure_path: Path,
    line_count: int,
    token_count: int,
) -> None:
    lines: list[str] = [
        "词向量实验运行结果",
        "=" * 24,
        f"语料行数: {line_count}",
        f"分词后 token 数: {token_count}",
        f"词表大小: {len(model.wv)}",
        "Word2Vec 参数: vector_size=100, window=5, min_count=1, workers=4, sg=1, epochs=5, seed=42",
        "",
    ]
    add_similarity_results(model, lines)
    add_nearest_results(model, lines)
    add_analogy_results(model, lines)
    run_pca(model, figure_path, lines)

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines), encoding="utf-8")
    print(results_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    segmented_path = DEFAULT_SEGMENTED_PATH.resolve()
    model_path = (output_dir / "models" / DEFAULT_MODEL_PATH.name).resolve()
    vectors_path = (output_dir / "vectors" / DEFAULT_VECTORS_PATH.name).resolve()
    results_path = (output_dir / "results" / DEFAULT_RESULTS_PATH.name).resolve()
    figure_path = (output_dir / "figures" / DEFAULT_FIGURE_PATH.name).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    segmented_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reuse_segmented and segmented_path.exists():
        print(f"Reusing segmented corpus: {segmented_path}")
        line_count = sum(1 for _ in segmented_path.open("r", encoding="utf-8"))
        token_count = sum(
            len(line.split()) for line in segmented_path.open("r", encoding="utf-8")
        )
    else:
        print(f"Segmenting corpus: {dataset_path}")
        line_count, token_count = segment_corpus(dataset_path, segmented_path)

    print("Training Word2Vec model...")
    model = train_model(segmented_path)
    model.save(str(model_path))
    model.wv.save_word2vec_format(str(vectors_path), binary=False)
    print(f"Saved model: {model_path}")
    print(f"Saved vectors: {vectors_path}")

    write_results(model, results_path, figure_path, line_count, token_count)


if __name__ == "__main__":
    main()
