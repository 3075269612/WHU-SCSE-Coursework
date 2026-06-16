"""
9.3.2 LSTM 模型案例：影评情感分析
=================================
对应第六次实验 LSTM 文本分类案例。教材使用 IMDB + TensorFlow/Keras；
本项目用小型内置影评数据和 NumPy 手写 LSTM 流程，保证离线可运行。
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# 教材原案例使用 IMDB 大型影评数据集和 TensorFlow/Keras。
# 项目指定 Python 环境没有 tensorflow，因此这里用小型内置影评数据复现同样流程:
# 文本向量化 -> 嵌入 -> 双向 LSTM 编码 -> 全连接二分类 -> 训练曲线与预测。
TRAIN_TEXTS = [
    ("The movie was cool and the animation was wonderful", 1),
    ("I would recommend this excellent film", 1),
    ("A touching story with great actors", 1),
    ("The graphics were out of this world", 1),
    ("This film is good and enjoyable", 1),
    ("The movie was not good and the plot was terrible", 0),
    ("I would not recommend this boring film", 0),
    ("Bad acting and awful graphics", 0),
    ("The story was dull and slow", 0),
    ("This movie is terrible and boring", 0),
]

TEST_TEXTS = [
    ("The movie was cool. I would recommend this movie.", 1),
    ("The animation and graphics were terrible. I would not recommend it.", 0),
    ("A good story with excellent actors", 1),
    ("A slow and awful movie", 0),
]

POSITIVE_WORDS = {"cool", "wonderful", "recommend", "excellent", "touching", "great", "good", "enjoyable"}
NEGATIVE_WORDS = {"not", "terrible", "boring", "bad", "awful", "dull", "slow"}


def tokenize(text):
    """把英文句子转成小写词元列表。

    这里保留英文单词和撇号，去掉标点符号，等价于最简单的文本标准化。
    """
    return re.findall(r"[a-z']+", text.lower())


class TextVectorization:
    """轻量复现 Keras TextVectorization: adapt 后把文本映射为词索引序列。

    <pad> 的编号固定为 0，用于补齐短句；
    <unk> 的编号固定为 1，用于表示词表外词。
    """

    def __init__(self, max_tokens=1000):
        self.max_tokens = max_tokens
        self.token_to_id = {"<pad>": 0, "<unk>": 1}

    def adapt(self, texts):
        """统计训练语料词频，并按词频构建词表。"""
        counts = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: self.max_tokens - 2]:
            self.token_to_id[token] = len(self.token_to_id)

    def get_vocabulary(self):
        """按编号顺序返回词表，便于观察前若干个高频词。"""
        vocab = [None] * len(self.token_to_id)
        for token, idx in self.token_to_id.items():
            vocab[idx] = token
        return vocab

    def __call__(self, texts, max_len=16):
        """把一批文本转换成固定长度整数矩阵。

        输出形状为 (batch_size, max_len)。长句截断，短句在尾部补 <pad>。
        """
        rows = []
        for text in texts:
            ids = [self.token_to_id.get(token, 1) for token in tokenize(text)]
            ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
            rows.append(ids)
        return np.array(rows, dtype=np.int64)


def sigmoid(x):
    """二分类输出层使用的 sigmoid 函数。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


class LSTMCell:
    """按 PDF 公式实现输入门、遗忘门、候选记忆元和输出门。

    i_t = sigmoid(x_t W_i + h_{t-1} U_i + b_i)
    f_t = sigmoid(x_t W_f + h_{t-1} U_f + b_f)
    o_t = sigmoid(x_t W_o + h_{t-1} U_o + b_o)
    c~_t = tanh(x_t W_c + h_{t-1} U_c + b_c)
    c_t = f_t * c_{t-1} + i_t * c~_t
    h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.w_i = rng.normal(0, 0.2, (input_dim, hidden_dim))
        self.u_i = rng.normal(0, 0.2, (hidden_dim, hidden_dim))
        self.b_i = np.ones(hidden_dim)
        self.w_f = rng.normal(0, 0.2, (input_dim, hidden_dim))
        self.u_f = rng.normal(0, 0.2, (hidden_dim, hidden_dim))
        self.b_f = np.ones(hidden_dim)
        self.w_o = rng.normal(0, 0.2, (input_dim, hidden_dim))
        self.u_o = rng.normal(0, 0.2, (hidden_dim, hidden_dim))
        self.b_o = np.ones(hidden_dim)
        self.w_c = rng.normal(0, 0.2, (input_dim, hidden_dim))
        self.u_c = rng.normal(0, 0.2, (hidden_dim, hidden_dim))
        self.b_c = np.zeros(hidden_dim)

    def __call__(self, sequence):
        # h 是隐藏状态，负责向下一层或输出层传递当前摘要；
        # c 是细胞状态，负责跨较长时间范围保存信息。
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        for x in sequence:
            i = sigmoid(x @ self.w_i + h @ self.u_i + self.b_i)
            f = sigmoid(x @ self.w_f + h @ self.u_f + self.b_f)
            o = sigmoid(x @ self.w_o + h @ self.u_o + self.b_o)
            c_tilde = np.tanh(x @ self.w_c + h @ self.u_c + self.b_c)
            c = f * c + i * c_tilde
            h = o * np.tanh(c)
        return h


class MiniBiLSTMClassifier:
    """双向 LSTM + Dense 的可运行小型情感分类器。

    为了在小数据集上得到稳定演示效果，嵌入向量第 0 维会注入一点先验:
    正面词为正值，负面词为负值。其余维度仍为随机初始化。
    """

    def __init__(self, vocab, embedding_dim=8, hidden_dim=8, seed=1):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        self.embeddings = rng.normal(0, 0.05, (len(vocab), embedding_dim))
        for token, idx in vocab.items():
            if token in POSITIVE_WORDS:
                self.embeddings[idx, 0] = 1.0
            if token in NEGATIVE_WORDS:
                self.embeddings[idx, 0] = -1.0
        self.forward_lstm = LSTMCell(embedding_dim, hidden_dim, seed=2)
        self.backward_lstm = LSTMCell(embedding_dim, hidden_dim, seed=3)
        self.w = rng.normal(0, 0.2, (hidden_dim * 2, 1))
        self.b = np.zeros(1)

    def encode(self, batch_ids):
        """把整数序列编码为句向量。

        前向 LSTM 从左到右读句子，后向 LSTM 从右到左读句子；
        两个最终隐藏状态拼接后作为整句表示。
        """
        features = []
        for ids in batch_ids:
            valid = [idx for idx in ids if idx != 0]
            if not valid:
                valid = [1]
            sequence = self.embeddings[valid]
            h_f = self.forward_lstm(sequence)
            h_b = self.backward_lstm(sequence[::-1])
            features.append(np.concatenate([h_f, h_b]))
        return np.vstack(features)

    def predict_logits(self, batch_ids):
        """输出未经过 sigmoid 的二分类分数 logits。"""
        return self.encode(batch_ids) @ self.w + self.b

    def fit_dense(self, batch_ids, labels, epochs=120, lr=0.4):
        """训练最后的 Dense 二分类层。

        本例重点演示 LSTM 门控编码流程，因此固定 LSTM 与嵌入层参数，
        只训练最后一层线性分类器，避免手写完整 BPTT 造成脚本过长。
        """
        labels = labels.reshape(-1, 1)
        features = self.encode(batch_ids)
        losses = []
        for _ in range(epochs):
            logits = features @ self.w + self.b
            probs = sigmoid(logits)
            loss = -np.mean(labels * np.log(probs + 1e-12) + (1 - labels) * np.log(1 - probs + 1e-12))
            grad = (probs - labels) / len(labels)
            self.w -= lr * (features.T @ grad)
            self.b -= lr * grad.sum(axis=0)
            losses.append(float(loss))
        return losses

    def predict(self, batch_ids):
        """把 sigmoid 概率 >= 0.5 的样本判为正面。"""
        return (sigmoid(self.predict_logits(batch_ids)).ravel() >= 0.5).astype(int)


def plot_graphs(losses, accuracy_values):
    """保存训练准确率和损失曲线。"""
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(accuracy_values)
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lstm_sentiment_training.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # 1. 构建词表，并展示词表前 20 项，对应教材中的 encoder.get_vocabulary()。
    encoder = TextVectorization(max_tokens=1000)
    encoder.adapt([text for text, _ in TRAIN_TEXTS])
    vocab = np.array(encoder.get_vocabulary())
    print("词表前 20 项:", vocab[:20].tolist())

    # 2. 把训练集和测试集文本转换为固定长度整数矩阵。
    x_train = encoder([text for text, _ in TRAIN_TEXTS])
    y_train = np.array([label for _, label in TRAIN_TEXTS])
    x_test = encoder([text for text, _ in TEST_TEXTS])
    y_test = np.array([label for _, label in TEST_TEXTS])
    print("编码示例:", x_train[:3])

    # 3. 构建双向 LSTM 情感分类器并训练 Dense 输出层。
    model = MiniBiLSTMClassifier(encoder.token_to_id)
    losses = []
    accuracy_values = []
    for epoch_loss in model.fit_dense(x_train, y_train, epochs=120, lr=0.4):
        losses.append(epoch_loss)
        accuracy_values.append(float(np.mean(model.predict(x_train) == y_train)))

    # 4. 在测试集上计算准确率，并对两条新句子做情感预测。
    test_pred = model.predict(x_test)
    print("Test Loss:", losses[-1])
    print("Test Accuracy:", float(np.mean(test_pred == y_test)))

    sample_text = "The movie was cool. The animation and the graphics were out of this world. I would recommend this movie."
    negative_text = "The movie was not good. The animation and the graphics were terrible. I would not recommend this movie."
    for text in [sample_text, negative_text]:
        score = sigmoid(model.predict_logits(encoder([text])))[0, 0]
        label = "正面" if score >= 0.5 else "负面"
        print(f"{label} {score:.3f}: {text}")

    print("堆叠 LSTM 说明: 中间 LSTM 需要 return_sequences=True，最后一层输出最终隐藏状态用于分类。")
    plot_graphs(losses, accuracy_values)
    print("训练曲线已保存: output/lstm_sentiment_training.png")
