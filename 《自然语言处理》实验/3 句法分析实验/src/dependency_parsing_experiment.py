from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw" / "句法分析数据集" / "依存分析训练数据"
EMBED_DIR = ROOT_DIR / "data" / "raw" / "embeddings"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

PAD = "<PAD>"
UNK = "<UNK>"
ROOT = "<ROOT>"


@dataclass
class Sentence:
    words: list[str]
    pos: list[str]
    heads: list[int]
    rels: list[str]


@dataclass
class RunConfig:
    name: str
    embed_dim: int
    embedding_mode: str
    optimizer: str
    embedding_file: str | None = None
    hidden_size: int = 96
    arc_dim: int = 96
    rel_dim: int = 64
    dropout: float = 0.25
    lr: float = 0.003


class BiaffineDependencyParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pos_size: int,
        rel_size: int,
        config: RunConfig,
        pretrained: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=0)
        if pretrained is not None:
            self.word_embedding.weight.data.copy_(torch.tensor(pretrained, dtype=torch.float32))
        self.pos_embedding = nn.Embedding(pos_size, 32, padding_idx=0)
        self.encoder = nn.LSTM(
            input_size=config.embed_dim + 32,
            hidden_size=config.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        encoder_dim = config.hidden_size * 2
        self.dropout = nn.Dropout(config.dropout)
        self.mlp_arc_dep = nn.Linear(encoder_dim, config.arc_dim)
        self.mlp_arc_head = nn.Linear(encoder_dim, config.arc_dim)
        self.mlp_rel_dep = nn.Linear(encoder_dim, config.rel_dim)
        self.mlp_rel_head = nn.Linear(encoder_dim, config.rel_dim)
        self.arc_weight = nn.Parameter(torch.empty(config.arc_dim + 1, config.arc_dim + 1))
        self.rel_weight = nn.Parameter(torch.empty(rel_size, config.rel_dim + 1, config.rel_dim + 1))
        nn.init.xavier_uniform_(self.arc_weight)
        nn.init.xavier_uniform_(self.rel_weight)

    @staticmethod
    def add_bias(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)

    def forward(self, words: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = torch.cat([self.word_embedding(words), self.pos_embedding(pos)], dim=-1)
        encoded, _ = self.encoder(self.dropout(embedded))
        encoded = self.dropout(encoded)
        arc_dep = self.add_bias(F.relu(self.mlp_arc_dep(encoded)))
        arc_head = self.add_bias(F.relu(self.mlp_arc_head(encoded)))
        rel_dep = self.add_bias(F.relu(self.mlp_rel_dep(encoded)))
        rel_head = self.add_bias(F.relu(self.mlp_rel_head(encoded)))
        arc_scores = torch.einsum("bxi,ij,byj->bxy", arc_dep, self.arc_weight, arc_head)
        return arc_scores, rel_dep, rel_head

    def rel_scores_for_heads(self, rel_dep: torch.Tensor, rel_head: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        selected_heads = rel_head.gather(1, heads.unsqueeze(-1).expand(-1, -1, rel_head.size(-1)))
        return torch.einsum("bxd,rdh,bxh->bxr", rel_dep, self.rel_weight, selected_heads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact Biaffine dependency parsing experiment.")
    parser.add_argument("--treebank", choices=("THU", "HIT"), default="THU")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train-limit", type=int, default=1200, help="0 means use the full train file.")
    parser.add_argument("--dev-limit", type=int, default=400, help="0 means use the full dev file.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_conll(path: Path, limit: int = 0) -> list[Sentence]:
    sentences: list[Sentence] = []
    words: list[str] = []
    pos: list[str] = []
    heads: list[int] = []
    rels: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line:
                if words:
                    sentences.append(Sentence(words, pos, heads, rels))
                    if limit and len(sentences) >= limit:
                        return sentences
                    words, pos, heads, rels = [], [], [], []
                continue
            parts = line.split("\t")
            if len(parts) < 8 or "-" in parts[0] or "." in parts[0]:
                continue
            words.append(parts[1])
            pos.append(parts[3])
            heads.append(int(parts[6]))
            rels.append(parts[7])
    if words and (not limit or len(sentences) < limit):
        sentences.append(Sentence(words, pos, heads, rels))
    return sentences


def build_vocab(items: list[str], min_freq: int = 1) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    vocab = {PAD: 0, UNK: 1, ROOT: 2}
    for item, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if count >= min_freq and item not in vocab:
            vocab[item] = len(vocab)
    return vocab


def build_label_vocab(items: list[str]) -> dict[str, int]:
    vocab = {PAD: 0, ROOT: 1, UNK: 2}
    for item in sorted(set(items)):
        if item not in vocab:
            vocab[item] = len(vocab)
    return vocab


def collate_batch(batch: list[Sentence], word_vocab: dict[str, int], pos_vocab: dict[str, int], rel_vocab: dict[str, int]):
    max_len = max(len(sentence.words) for sentence in batch) + 1
    word_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    pos_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    heads = torch.zeros(len(batch), max_len, dtype=torch.long)
    rels = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for row, sentence in enumerate(batch):
        length = len(sentence.words) + 1
        word_ids[row, 0] = word_vocab[ROOT]
        pos_ids[row, 0] = pos_vocab[ROOT]
        rels[row, 0] = rel_vocab[ROOT]
        mask[row, :length] = True
        for idx, (word, tag, head, rel) in enumerate(zip(sentence.words, sentence.pos, sentence.heads, sentence.rels), start=1):
            word_ids[row, idx] = word_vocab.get(word, word_vocab[UNK])
            pos_ids[row, idx] = pos_vocab.get(tag, pos_vocab[UNK])
            heads[row, idx] = head
            rels[row, idx] = rel_vocab.get(rel, rel_vocab[UNK])
    return word_ids, pos_ids, heads, rels, mask


def make_loader(data: list[Sentence], batch_size: int, shuffle: bool, seed: int, word_vocab, pos_vocab, rel_vocab) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=lambda batch: collate_batch(batch, word_vocab, pos_vocab, rel_vocab),
    )


def load_embedding_file(path: Path) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, line in enumerate(file):
            parts = line.strip().split()
            if line_number == 0 and len(parts) == 2 and all(part.isdigit() for part in parts):
                continue
            if len(parts) < 3:
                continue
            try:
                vector = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            vectors[parts[0]] = vector
    return vectors


def build_pretrained_matrix(word_vocab: dict[str, int], config: RunConfig) -> tuple[np.ndarray | None, int, str]:
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.05, size=(len(word_vocab), config.embed_dim)).astype(np.float32)
    matrix[0] = 0
    if config.embedding_mode == "random":
        return matrix, 0, "random"
    if not config.embedding_file:
        return matrix, 0, "missing"
    vectors = load_embedding_file(EMBED_DIR / config.embedding_file)
    hits = 0
    for word, idx in word_vocab.items():
        if word in (PAD, UNK, ROOT):
            continue
        vector = vectors.get(word)
        source = "exact"
        if vector is None and config.embedding_mode == "char_average":
            char_vectors = [vectors[char] for char in word if char in vectors and len(vectors[char]) == config.embed_dim]
            if char_vectors:
                vector = np.mean(char_vectors, axis=0)
                source = "char_average"
        if vector is not None and len(vector) == config.embed_dim:
            matrix[idx] = vector
            hits += 1
    return matrix, hits, f"{config.embedding_file}:{config.embedding_mode}"


def mask_arc_scores(arc_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = arc_scores.clone()
    scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
    length = scores.size(1)
    eye = torch.eye(length, dtype=torch.bool, device=scores.device).unsqueeze(0)
    return scores.masked_fill(eye, -1e9)


def valid_dependency_mask(mask: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
    valid = mask.clone()
    valid[:, 0] = False
    positions = torch.arange(mask.size(1), device=mask.device).unsqueeze(0)
    valid &= heads != positions
    return valid


def compute_loss(model: BiaffineDependencyParser, batch, device: torch.device) -> tuple[torch.Tensor, int]:
    words, pos, heads, rels, mask = [item.to(device) for item in batch]
    arc_scores, rel_dep, rel_head = model(words, pos)
    arc_scores = mask_arc_scores(arc_scores, mask)
    valid = valid_dependency_mask(mask, heads)
    arc_loss = F.cross_entropy(arc_scores[valid], heads[valid])
    rel_scores = model.rel_scores_for_heads(rel_dep, rel_head, heads.clamp_min(0))
    rel_loss = F.cross_entropy(rel_scores[valid], rels[valid])
    return arc_loss + rel_loss, int(valid.sum().item())


def evaluate(model: BiaffineDependencyParser, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    arc_correct = 0
    rel_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            words, pos, heads, rels, mask = [item.to(device) for item in batch]
            arc_scores, rel_dep, rel_head = model(words, pos)
            pred_heads = mask_arc_scores(arc_scores, mask).argmax(dim=-1)
            rel_scores = model.rel_scores_for_heads(rel_dep, rel_head, pred_heads)
            pred_rels = rel_scores.argmax(dim=-1)
            valid = valid_dependency_mask(mask, heads)
            correct_arc = pred_heads[valid] == heads[valid]
            arc_correct += int(correct_arc.sum().item())
            rel_correct += int((correct_arc & (pred_rels[valid] == rels[valid])).sum().item())
            total += int(valid.sum().item())
    return arc_correct / total, rel_correct / total


def train_one(config: RunConfig, train_data, dev_data, vocabs, args, device: torch.device) -> dict:
    word_vocab, pos_vocab, rel_vocab = vocabs
    set_seed(args.seed)
    pretrained, hits, embed_source = build_pretrained_matrix(word_vocab, config)
    model = BiaffineDependencyParser(len(word_vocab), len(pos_vocab), len(rel_vocab), config, pretrained).to(device)
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    else:
        raise ValueError(config.optimizer)
    train_loader = make_loader(train_data, args.batch_size, True, args.seed, word_vocab, pos_vocab, rel_vocab)
    dev_loader = make_loader(dev_data, args.batch_size, False, args.seed, word_vocab, pos_vocab, rel_vocab)
    history: list[dict] = []
    best_las = -1.0
    best_state = None
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        token_total = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss, token_count = compute_loss(model, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.item()) * token_count
            token_total += token_count
        uas, las = evaluate(model, dev_loader, device)
        avg_loss = loss_total / token_total
        history.append({"epoch": epoch, "train_loss": avg_loss, "dev_uas": uas, "dev_las": las})
        if las > best_las:
            best_las = las
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{config.name} epoch={epoch} loss={avg_loss:.4f} dev_uas={uas:.4f} dev_las={las:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    uas, las = evaluate(model, dev_loader, device)
    return {
        "config": config,
        "history": history,
        "dev_uas": uas,
        "dev_las": las,
        "embedding_hits": hits,
        "embedding_source": embed_source,
        "elapsed_seconds": time.time() - start,
        "state": best_state,
        "model": model,
    }


def experiment_configs() -> list[RunConfig]:
    return [
        RunConfig("random_100_adam", 100, "random", "adam", lr=0.003),
        RunConfig("random_300_adam", 300, "random", "adam", lr=0.003),
        RunConfig("giga_sample_100_adam", 100, "exact", "adam", "giga.100.txt.sample", lr=0.003),
        RunConfig("charavg_100_adam", 100, "char_average", "adam", "char_word2vec.dat", lr=0.003),
        RunConfig("random_100_sgd", 100, "random", "sgd", lr=0.05),
        RunConfig("charavg_100_sgd", 100, "char_average", "sgd", "char_word2vec.dat", lr=0.05),
    ]


def write_outputs(results: list[dict], train_data, dev_data, vocabs, args, device: torch.device) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "metrics.csv"
    history_path = RESULTS_DIR / "training_history.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "embed_dim", "embedding", "optimizer", "embedding_hits", "dev_uas", "dev_las", "elapsed_seconds"])
        for result in results:
            config = result["config"]
            writer.writerow(
                [
                    config.name,
                    config.embed_dim,
                    result["embedding_source"],
                    config.optimizer,
                    result["embedding_hits"],
                    f"{result['dev_uas']:.6f}",
                    f"{result['dev_las']:.6f}",
                    f"{result['elapsed_seconds']:.2f}",
                ]
            )
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "epoch", "train_loss", "dev_uas", "dev_las"])
        for result in results:
            for row in result["history"]:
                writer.writerow([result["config"].name, row["epoch"], f"{row['train_loss']:.6f}", f"{row['dev_uas']:.6f}", f"{row['dev_las']:.6f}"])

    best = max(results, key=lambda item: item["dev_las"])
    torch.save(best["state"], MODELS_DIR / "best_biaffine_parser.pt")
    word_vocab, pos_vocab, rel_vocab = vocabs
    (PROCESSED_DIR / "word_vocab.json").write_text(json.dumps(word_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "pos_vocab.json").write_text(json.dumps(pos_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "rel_vocab.json").write_text(json.dumps(rel_vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_loss(results)
    plot_metrics(results)
    write_sample_parse(best, dev_data[0], vocabs, device)
    write_summary(results, train_data, dev_data, args, best)


def plot_loss(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        epochs = [row["epoch"] for row in result["history"]]
        losses = [row["train_loss"] for row in result["history"]]
        ax.plot(epochs, losses, marker="o", label=result["config"].name)
    ax.set_title("Training loss for Biaffine dependency parsing")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_loss.png", dpi=180)
    plt.close(fig)


def plot_metrics(results: list[dict]) -> None:
    names = [result["config"].name for result in results]
    uas = [result["dev_uas"] * 100 for result in results]
    las = [result["dev_las"] * 100 for result in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, uas, width=0.36, label="UAS")
    ax.bar(x + 0.18, las, width=0.36, label="LAS")
    ax.set_ylabel("score (%)")
    ax.set_title("Development UAS/LAS by embedding and optimizer")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "uas_las_comparison.png", dpi=180)
    plt.close(fig)


def predict_sentence(result: dict, sentence: Sentence, vocabs, device: torch.device) -> tuple[list[int], list[str]]:
    word_vocab, pos_vocab, rel_vocab = vocabs
    inv_rel = {idx: rel for rel, idx in rel_vocab.items()}
    model = result["model"]
    model.eval()
    batch = collate_batch([sentence], word_vocab, pos_vocab, rel_vocab)
    words, pos, _, _, mask = [item.to(device) for item in batch]
    with torch.no_grad():
        arc_scores, rel_dep, rel_head = model(words, pos)
        pred_heads = mask_arc_scores(arc_scores, mask).argmax(dim=-1)
        pred_rels = model.rel_scores_for_heads(rel_dep, rel_head, pred_heads).argmax(dim=-1)
    heads = pred_heads[0, 1 : len(sentence.words) + 1].cpu().tolist()
    rels = [inv_rel.get(idx, UNK) for idx in pred_rels[0, 1 : len(sentence.words) + 1].cpu().tolist()]
    return heads, rels


def write_sample_parse(best: dict, sentence: Sentence, vocabs, device: torch.device) -> None:
    pred_heads, pred_rels = predict_sentence(best, sentence, vocabs, device)
    lines = ["ID\tWORD\tGOLD_HEAD\tGOLD_REL\tPRED_HEAD\tPRED_REL"]
    for idx, (word, gold_head, gold_rel, pred_head, pred_rel) in enumerate(
        zip(sentence.words, sentence.heads, sentence.rels, pred_heads, pred_rels), start=1
    ):
        lines.append(f"{idx}\t{word}\t{gold_head}\t{gold_rel}\t{pred_head}\t{pred_rel}")
    (RESULTS_DIR / "sample_parse.txt").write_text("\n".join(lines), encoding="utf-8")
    plot_dependency_tree(sentence.words, pred_heads, pred_rels)


def plot_dependency_tree(words: list[str], heads: list[int], rels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(words) * 0.55), 4.0))
    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(-0.7, 4.0)
    ax.axis("off")
    for idx, word in enumerate(words):
        ax.text(idx, 0, word, ha="center", va="center", fontsize=10)
        ax.text(idx, -0.35, str(idx + 1), ha="center", va="center", fontsize=8, color="gray")
    for dep_idx, (head, rel) in enumerate(zip(heads, rels), start=1):
        if head == 0:
            ax.text(dep_idx - 1, 3.6, "ROOT", ha="center", fontsize=8, color="#9B1C1C")
            ax.annotate("", xy=(dep_idx - 1, 0.25), xytext=(dep_idx - 1, 3.35), arrowprops={"arrowstyle": "->", "lw": 1.1})
            continue
        start = head - 1
        end = dep_idx - 1
        mid = (start + end) / 2
        height = min(3.2, 0.7 + abs(end - start) * 0.22)
        ax.annotate("", xy=(end, 0.25), xytext=(start, 0.25), arrowprops={"arrowstyle": "->", "connectionstyle": f"arc3,rad={0.35 if end > start else -0.35}", "lw": 1.0})
        ax.text(mid, height, rel[:8], ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sample_dependency_tree.png", dpi=180)
    plt.close(fig)


def write_summary(results: list[dict], train_data, dev_data, args, best) -> None:
    lines = ["句法分析实验结果汇总", "=" * 40]
    lines.append(f"树库: {args.treebank}")
    lines.append(f"训练句数: {len(train_data)}, 开发句数: {len(dev_data)}, epochs: {args.epochs}, batch_size: {args.batch_size}")
    lines.append("")
    lines.append("配置\t维度\t词向量/初始化\t优化器\t命中词数\tUAS\tLAS\t耗时")
    for result in results:
        config = result["config"]
        lines.append(
            f"{config.name}\t{config.embed_dim}\t{result['embedding_source']}\t{config.optimizer}\t"
            f"{result['embedding_hits']}\t{result['dev_uas']:.4f}\t{result['dev_las']:.4f}\t{result['elapsed_seconds']:.2f}s"
        )
    lines.append("")
    lines.append(
        f"最佳配置: {best['config'].name}, UAS={best['dev_uas']:.4f}, LAS={best['dev_las']:.4f}, "
        f"embedding={best['embedding_source']}, optimizer={best['config'].optimizer}"
    )
    (RESULTS_DIR / "experiment_results.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    for directory in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = read_conll(DATA_DIR / args.treebank / "train.conll", args.train_limit)
    dev_data = read_conll(DATA_DIR / args.treebank / "dev.conll", args.dev_limit)
    all_words = [word for sentence in train_data for word in sentence.words]
    all_pos = [tag for sentence in train_data + dev_data for tag in sentence.pos]
    all_rels = [rel for sentence in train_data + dev_data for rel in sentence.rels]
    word_vocab = build_vocab(all_words, min_freq=1)
    pos_vocab = build_label_vocab(all_pos)
    rel_vocab = build_label_vocab(all_rels)
    vocabs = (word_vocab, pos_vocab, rel_vocab)

    results: list[dict] = []
    for config in experiment_configs():
        print(f"Running {config.name}")
        results.append(train_one(config, train_data, dev_data, vocabs, args, device))
    write_outputs(results, train_data, dev_data, vocabs, args, device)
    print(RESULTS_DIR / "experiment_results.txt")


if __name__ == "__main__":
    main()
