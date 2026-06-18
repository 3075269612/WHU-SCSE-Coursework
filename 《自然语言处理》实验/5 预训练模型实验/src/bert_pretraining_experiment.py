from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
MODELS_DIR = ROOT_DIR / "outputs" / "models"

DIALOGUE = (
    "Hello, how are you? I am Romeo.\n"
    "Hello, Romeo My name is Juliet. Nice to meet you.\n"
    "Nice meet you too. How are you today?\n"
    "Great. My baseball team won the competition.\n"
    "Oh Congratulations, Juliet\n"
    "Thank you Romeo\n"
    "Where are you going today?\n"
    "I am going shopping. What about you?\n"
    "I am going to visit my grandmother. She is not very well."
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    epochs: int
    learning_rate: float = 0.002
    batch_size: int = 6
    max_len: int = 30
    max_pred: int = 5
    dropout: float = 0.1
    seed: int = 42

    @property
    def d_k(self) -> int:
        return self.d_model // self.n_heads


def ensure_directories() -> None:
    for directory in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def preprocess_dialogue(text: str) -> list[str]:
    translation = str.maketrans("", "", ".,?!")
    return [
        line.translate(translation).lower().strip()
        for line in text.splitlines()
        if line.strip()
    ]


def build_vocabulary(sentences: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    words = sorted(set(" ".join(sentences).split()))
    word2idx = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3}
    word2idx.update({word: index + 4 for index, word in enumerate(words)})
    idx2word = {index: word for word, index in word2idx.items()}
    return word2idx, idx2word


def encode_sentences(sentences: list[str], word2idx: dict[str, int]) -> list[list[int]]:
    return [[word2idx[word] for word in sentence.split()] for sentence in sentences]


def mask_tokens(
    input_ids: list[int],
    word2idx: dict[str, int],
    vocab_size: int,
    max_pred: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    candidate_positions = [
        index
        for index, token in enumerate(input_ids)
        if token not in (word2idx["[CLS]"], word2idx["[SEP]"])
    ]
    rng.shuffle(candidate_positions)
    n_pred = min(max_pred, max(1, round(len(input_ids) * 0.15)))
    masked_tokens: list[int] = []
    masked_positions: list[int] = []

    for position in candidate_positions[:n_pred]:
        masked_positions.append(position)
        masked_tokens.append(input_ids[position])
        draw = rng.random()
        if draw < 0.8:
            input_ids[position] = word2idx["[MASK]"]
        elif draw < 0.9:
            input_ids[position] = rng.randrange(4, vocab_size)
        # The remaining 10% keep the original token.

    return input_ids, masked_tokens, masked_positions


def make_pretraining_batch(
    token_list: list[list[int]],
    word2idx: dict[str, int],
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    rng = random.Random(config.seed)
    positive = negative = 0
    target_per_class = config.batch_size // 2
    batch: list[dict[str, object]] = []

    while positive < target_per_class or negative < target_per_class:
        index_a = rng.randrange(len(token_list))
        index_b = rng.randrange(len(token_list))
        is_next = index_a + 1 == index_b
        if is_next and positive >= target_per_class:
            continue
        if not is_next and negative >= target_per_class:
            continue

        tokens_a = token_list[index_a]
        tokens_b = token_list[index_b]
        input_ids = (
            [word2idx["[CLS]"]]
            + tokens_a
            + [word2idx["[SEP]"]]
            + tokens_b
            + [word2idx["[SEP]"]]
        )
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = input_ids[: config.max_len]
        segment_ids = segment_ids[: config.max_len]
        input_ids, masked_tokens, masked_positions = mask_tokens(
            input_ids,
            word2idx,
            len(word2idx),
            config.max_pred,
            rng,
        )

        input_padding = config.max_len - len(input_ids)
        pred_padding = config.max_pred - len(masked_tokens)
        input_ids.extend([word2idx["[PAD]"]] * input_padding)
        segment_ids.extend([0] * input_padding)
        masked_tokens.extend([word2idx["[PAD]"]] * pred_padding)
        masked_positions.extend([0] * pred_padding)

        batch.append(
            {
                "sentence_a_index": index_a,
                "sentence_b_index": index_b,
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "masked_tokens": masked_tokens,
                "masked_positions": masked_positions,
                "is_next": int(is_next),
            }
        )
        positive += int(is_next)
        negative += int(not is_next)

    return batch


class PretrainingDataset(Dataset):
    def __init__(self, batch: list[dict[str, object]]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.batch[index]
        return (
            torch.tensor(sample["input_ids"], dtype=torch.long),
            torch.tensor(sample["segment_ids"], dtype=torch.long),
            torch.tensor(sample["masked_tokens"], dtype=torch.long),
            torch.tensor(sample["masked_positions"], dtype=torch.long),
            torch.tensor(sample["is_next"], dtype=torch.long),
        )


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, config.d_model, padding_idx=0)
        self.position = nn.Embedding(config.max_len, config.d_model)
        self.segment = nn.Embedding(2, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        positions = positions.unsqueeze(0).expand_as(input_ids)
        embeddings = (
            self.token(input_ids)
            + self.position(positions)
            + self.segment(segment_ids)
        )
        return self.dropout(self.norm(embeddings))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        projection_size = config.n_heads * config.d_k
        self.q_proj = nn.Linear(config.d_model, projection_size)
        self.k_proj = nn.Linear(config.d_model, projection_size)
        self.v_proj = nn.Linear(config.d_model, projection_size)
        self.out_proj = nn.Linear(projection_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        query = split_heads(self.q_proj(hidden))
        key = split_heads(self.k_proj(hidden))
        value = split_heads(self.v_proj(hidden))
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(pad_mask[:, None, None, :], -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attention), value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(context), attention


class EncoderLayer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(
        self, hidden: torch.Tensor, pad_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention = self.attention(hidden, pad_mask)
        hidden = self.norm1(hidden + self.dropout1(attention_output))
        feed_forward_output = self.feed_forward(hidden)
        hidden = self.norm2(hidden + self.dropout2(feed_forward_output))
        return hidden, attention


class MiniBertForPretraining(nn.Module):
    def __init__(self, vocab_size: int, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = BertEmbedding(vocab_size, config)
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.pooler = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
        )
        self.nsp_classifier = nn.Linear(config.d_model, 2)
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.mlm_decoder = nn.Linear(config.d_model, vocab_size)
        self.mlm_decoder.weight = self.embedding.token.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_mask = input_ids.eq(0)
        hidden = self.embedding(input_ids, segment_ids)
        attention = torch.empty(0, device=input_ids.device)
        for layer in self.layers:
            hidden, attention = layer(hidden, pad_mask)

        pooled = self.pooler(hidden[:, 0])
        nsp_logits = self.nsp_classifier(pooled)
        gather_index = masked_positions.unsqueeze(-1).expand(
            -1, -1, self.config.d_model
        )
        masked_hidden = torch.gather(hidden, 1, gather_index)
        mlm_logits = self.mlm_decoder(self.mlm_transform(masked_hidden))
        return mlm_logits, nsp_logits, attention


def calculate_metrics(
    mlm_logits: torch.Tensor,
    nsp_logits: torch.Tensor,
    masked_tokens: torch.Tensor,
    is_next: torch.Tensor,
) -> tuple[float, float]:
    mlm_predictions = mlm_logits.argmax(dim=-1)
    valid = masked_tokens.ne(0)
    mlm_accuracy = (
        mlm_predictions.eq(masked_tokens).logical_and(valid).sum().item()
        / max(1, valid.sum().item())
    )
    nsp_accuracy = nsp_logits.argmax(dim=-1).eq(is_next).float().mean().item()
    return mlm_accuracy, nsp_accuracy


def train_experiment(
    config: ExperimentConfig,
    batch: list[dict[str, object]],
    vocab_size: int,
) -> tuple[MiniBertForPretraining, list[dict[str, float]], dict[str, float]]:
    set_seed(config.seed)
    dataset = PretrainingDataset(batch)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = MiniBertForPretraining(vocab_size, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=0)
    nsp_criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        for input_ids, segment_ids, masked_tokens, masked_positions, is_next in loader:
            mlm_logits, nsp_logits, _ = model(
                input_ids, segment_ids, masked_positions
            )
            mlm_loss = mlm_criterion(
                mlm_logits.reshape(-1, vocab_size), masked_tokens.reshape(-1)
            )
            nsp_loss = nsp_criterion(nsp_logits, is_next)
            loss = mlm_loss + nsp_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            all_items = tuple(
                torch.stack([dataset[index][field] for index in range(len(dataset))])
                for field in range(5)
            )
            mlm_logits, nsp_logits, _ = model(
                all_items[0], all_items[1], all_items[3]
            )
            eval_mlm_loss = mlm_criterion(
                mlm_logits.reshape(-1, vocab_size), all_items[2].reshape(-1)
            )
            eval_nsp_loss = nsp_criterion(nsp_logits, all_items[4])
            eval_loss = eval_mlm_loss + eval_nsp_loss
            mlm_accuracy, nsp_accuracy = calculate_metrics(
                mlm_logits, nsp_logits, all_items[2], all_items[4]
            )
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(eval_loss.item()),
                "mlm_loss": float(eval_mlm_loss.item()),
                "nsp_loss": float(eval_nsp_loss.item()),
                "mlm_accuracy": mlm_accuracy,
                "nsp_accuracy": nsp_accuracy,
            }
        )
        if epoch == 1 or epoch % max(1, config.epochs // 5) == 0:
            print(
                f"[{config.name}] epoch={epoch:03d} "
                f"loss={eval_loss.item():.4f} "
                f"mlm_acc={mlm_accuracy:.3f} nsp_acc={nsp_accuracy:.3f}"
            )

    elapsed = time.perf_counter() - started
    final = history[-1]
    first_perfect_epoch = next(
        (
            int(row["epoch"])
            for row in history
            if row["mlm_accuracy"] == 1.0 and row["nsp_accuracy"] == 1.0
        ),
        config.epochs,
    )
    summary = {
        "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
        "elapsed_seconds": elapsed,
        "first_perfect_epoch": float(first_perfect_epoch),
        **{key: value for key, value in final.items() if key != "epoch"},
    }
    return model, history, summary


def predict_sample(
    model: MiniBertForPretraining,
    sample: dict[str, object],
    idx2word: dict[int, str],
) -> dict[str, object]:
    model.eval()
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long)
    segment_ids = torch.tensor([sample["segment_ids"]], dtype=torch.long)
    masked_positions = torch.tensor([sample["masked_positions"]], dtype=torch.long)
    with torch.no_grad():
        mlm_logits, nsp_logits, attention = model(
            input_ids, segment_ids, masked_positions
        )
    predicted_ids = mlm_logits.argmax(dim=-1)[0].tolist()
    target_ids = sample["masked_tokens"]
    valid_pairs = [
        (idx2word[int(target)], idx2word[int(prediction)])
        for target, prediction in zip(target_ids, predicted_ids)
        if int(target) != 0
    ]
    visible_tokens = [
        idx2word[int(token)] for token in sample["input_ids"] if int(token) != 0
    ]
    return {
        "input_tokens": visible_tokens,
        "masked_targets": [target for target, _ in valid_pairs],
        "masked_predictions": [prediction for _, prediction in valid_pairs],
        "is_next_target": bool(sample["is_next"]),
        "is_next_prediction": bool(nsp_logits.argmax(dim=-1).item()),
        "attention_shape": list(attention.shape),
    }


def write_outputs(
    configs: list[ExperimentConfig],
    histories: dict[str, list[dict[str, float]]],
    summaries: dict[str, dict[str, float]],
    prediction: dict[str, object],
    word2idx: dict[str, int],
    batch: list[dict[str, object]],
) -> None:
    (PROCESSED_DIR / "vocabulary.json").write_text(
        json.dumps(word2idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PROCESSED_DIR / "pretraining_batch.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "training_history.json").write_text(
        json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "sample_prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (RESULTS_DIR / "parameter_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "name",
                "d_model",
                "n_layers",
                "n_heads",
                "d_ff",
                "epochs",
                "parameters",
                "elapsed_seconds",
                "first_perfect_epoch",
                "loss",
                "mlm_loss",
                "nsp_loss",
                "mlm_accuracy",
                "nsp_accuracy",
            ]
        )
        for config in configs:
            summary = summaries[config.name]
            writer.writerow(
                [
                    config.name,
                    config.d_model,
                    config.n_layers,
                    config.n_heads,
                    config.d_ff,
                    config.epochs,
                    int(summary["parameters"]),
                    f"{summary['elapsed_seconds']:.4f}",
                    int(summary["first_perfect_epoch"]),
                    f"{summary['loss']:.6f}",
                    f"{summary['mlm_loss']:.6f}",
                    f"{summary['nsp_loss']:.6f}",
                    f"{summary['mlm_accuracy']:.6f}",
                    f"{summary['nsp_accuracy']:.6f}",
                ]
            )

    lines = [
        "预训练模型实验运行结果",
        "=" * 32,
        f"对话句子数: {len(preprocess_dialogue(DIALOGUE))}",
        f"词表大小: {len(word2idx)}",
        f"预训练样本数: {len(batch)}（正负 NSP 样本各 {len(batch) // 2}）",
        "",
        "一、参数对比",
    ]
    for config in configs:
        summary = summaries[config.name]
        lines.extend(
            [
                f"[{config.name}] d_model={config.d_model}, layers={config.n_layers}, "
                f"heads={config.n_heads}, d_ff={config.d_ff}, epochs={config.epochs}",
                f"  参数量: {int(summary['parameters']):,}",
                f"  耗时: {summary['elapsed_seconds']:.3f} 秒",
                f"  首次 MLM/NSP 同时达到 100% 的轮次: "
                f"{int(summary['first_perfect_epoch'])}",
                f"  最终总损失: {summary['loss']:.6f}",
                f"  MLM 损失/准确率: {summary['mlm_loss']:.6f} / "
                f"{summary['mlm_accuracy']:.4f}",
                f"  NSP 损失/准确率: {summary['nsp_loss']:.6f} / "
                f"{summary['nsp_accuracy']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "二、基线模型测试样例",
            f"输入 token: {' '.join(prediction['input_tokens'])}",
            f"被掩码真实词: {prediction['masked_targets']}",
            f"被掩码预测词: {prediction['masked_predictions']}",
            f"NSP 真实标签: {prediction['is_next_target']}",
            f"NSP 预测标签: {prediction['is_next_prediction']}",
            f"末层注意力张量形状: {prediction['attention_shape']}",
        ]
    )
    (RESULTS_DIR / "experiment_results.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (RESULTS_DIR / "experiment_config.json").write_text(
        json.dumps([asdict(config) for config in configs], indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact BERT model on MLM and NSP tasks."
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run 30 epochs per configuration for a fast smoke test.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip parameter comparison and train only the baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    epochs = 30 if args.quick else args.epochs
    sentences = preprocess_dialogue(DIALOGUE)
    word2idx, idx2word = build_vocabulary(sentences)
    token_list = encode_sentences(sentences, word2idx)

    configs = [
        ExperimentConfig(
            name="baseline",
            d_model=128,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            epochs=epochs,
        )
    ]
    if not args.baseline_only:
        configs.extend(
            [
                ExperimentConfig(
                    name="shallow",
                    d_model=128,
                    n_layers=1,
                    n_heads=4,
                    d_ff=256,
                    epochs=epochs,
                ),
                ExperimentConfig(
                    name="narrow",
                    d_model=64,
                    n_layers=2,
                    n_heads=4,
                    d_ff=128,
                    epochs=epochs,
                ),
            ]
        )

    reference_batch = make_pretraining_batch(token_list, word2idx, configs[0])
    histories: dict[str, list[dict[str, float]]] = {}
    summaries: dict[str, dict[str, float]] = {}
    baseline_model: MiniBertForPretraining | None = None

    for config in configs:
        batch = make_pretraining_batch(token_list, word2idx, config)
        model, history, summary = train_experiment(config, batch, len(word2idx))
        histories[config.name] = history
        summaries[config.name] = summary
        if config.name == "baseline":
            baseline_model = model
            model_path = MODELS_DIR / "mini_bert_pretraining.pt"
            # Passing a file object avoids a Windows/PyTorch Unicode-path issue.
            with model_path.open("wb") as model_file:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "word2idx": word2idx,
                    },
                    model_file,
                )

    assert baseline_model is not None
    prediction = predict_sample(baseline_model, reference_batch[0], idx2word)
    write_outputs(
        configs,
        histories,
        summaries,
        prediction,
        word2idx,
        reference_batch,
    )
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "src" / "plot_experiment_results.py")],
        check=True,
    )
    print(f"Results written to: {RESULTS_DIR}")
    print(f"Figures written to: {FIGURES_DIR}")
    print(f"Model written to: {MODELS_DIR / 'mini_bert_pretraining.pt'}")


if __name__ == "__main__":
    main()
