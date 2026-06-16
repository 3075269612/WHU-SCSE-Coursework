"""
10.3.1 基于 GCN 的节点分类
==========================
对应第六次实验 Cora 节点分类案例。为保证离线可运行，本脚本使用
NumPy/SciPy 构造 Cora 风格小图并手写两层 GCN 训练流程。
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def normalization(adjacency):
    """GCN 标准归一化: L = D^-0.5 * (A + I) * D^-0.5。"""
    adjacency = adjacency.astype(np.float32) + sp.eye(adjacency.shape[0], dtype=np.float32)
    degree = np.array(adjacency.sum(1)).flatten()
    degree[degree == 0] = 1.0
    d_hat = sp.diags(np.power(degree, -0.5))
    return d_hat.dot(adjacency).dot(d_hat).tocsr()


def make_cora_like_graph(num_classes=7, nodes_per_class=20, feature_dim=28, seed=0):
    """构造一个 Cora 风格的节点分类小图。

    教材案例依赖 torch_geometric 自动下载 Cora 数据集，但项目规定环境没有
    torch_geometric。为了让脚本可运行，这里构造一个小型替代图:
        - 7 个类别，对应 Cora 的 7 分类任务。
        - 每类 20 个节点。
        - 同类节点特征围绕同一个 prototype 采样。
        - 同类内部连接较密，同时加入少量跨类随机边。
    """
    rng = np.random.default_rng(seed)
    num_nodes = num_classes * nodes_per_class
    # labels 是节点真实类别，形状为 (num_nodes,)。
    labels = np.repeat(np.arange(num_classes), nodes_per_class)

    # 每个类别生成一个特征原型；同类节点在原型附近加噪声，模拟论文主题词向量。
    prototypes = rng.normal(0, 1, (num_classes, feature_dim))
    features = prototypes[labels] + rng.normal(0, 0.45, (num_nodes, feature_dim))
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)

    rows, cols = [], []
    for c in range(num_classes):
        start = c * nodes_per_class
        for i in range(nodes_per_class):
            a = start + i
            for offset in (1, 2):
                # 同类节点连成环并连接近邻，保证图结构携带类别信息。
                b = start + (i + offset) % nodes_per_class
                rows.extend([a, b])
                cols.extend([b, a])
    for _ in range(num_classes * 8):
        # 加入少量跨类随机边，模拟真实引用网络中的噪声连接。
        a, b = rng.choice(num_nodes, 2, replace=False)
        rows.extend([a, b])
        cols.extend([b, a])
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

    train_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        # 每类前 5 个节点作为训练节点，其余作为测试节点。
        train_mask[idx[:5]] = True
        test_mask[idx[5:]] = True
    return features.astype(np.float32), labels, adjacency, train_mask, test_mask


def relu(x):
    """ReLU 激活函数。"""
    return np.maximum(0, x)


def softmax(x):
    """按行计算 softmax，得到每个节点属于各类别的概率。"""
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


class NumpyGCN:
    """两层 GCN 节点分类模型。

    结构与 PyG 示例中的 GCNConv -> ReLU -> GCNConv 一致:
        H = ReLU(L X W1)
        logits = L H W2
    这里用 NumPy/SciPy 手写，避免依赖 torch_geometric。
    """

    def __init__(self, num_features, hidden_channels, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.2, (num_features, hidden_channels))
        self.w2 = rng.normal(0, 0.2, (hidden_channels, num_classes))

    def forward(self, adjacency_norm, x):
        """前向传播，并返回反向传播需要的中间结果。"""
        # 第 1 层先聚合邻居原始特征，再做线性变换。
        ax = adjacency_norm @ x
        z1 = ax @ self.w1
        h = relu(z1)
        # 第 2 层聚合隐藏表示，再映射到 7 个类别的 logits。
        ah = adjacency_norm @ h
        logits = ah @ self.w2
        cache = (ax, z1, h, ah)
        return logits, cache

    def train_step(self, adjacency_norm, x, y, train_mask, lr=0.05, weight_decay=5e-4):
        """执行一个 epoch 的梯度下降。

        损失只在 train_mask 指定的训练节点上计算；
        test_mask 节点不参与训练，只用于最后评估。
        """
        logits, cache = self.forward(adjacency_norm, x)
        probs = softmax(logits)
        train_idx = np.where(train_mask)[0]
        loss = -np.mean(np.log(probs[train_idx, y[train_idx]] + 1e-12))
        loss += weight_decay * (np.sum(self.w1 * self.w1) + np.sum(self.w2 * self.w2))

        # 交叉熵对 logits 的梯度: softmax(logits) - one_hot(label)。
        dlogits = np.zeros_like(probs)
        dlogits[train_idx] = probs[train_idx]
        dlogits[train_idx, y[train_idx]] -= 1.0
        dlogits[train_idx] /= len(train_idx)

        # 手写反向传播，依次求 W2 和 W1 的梯度。
        ax, z1, h, ah = cache
        dw2 = ah.T @ dlogits + 2 * weight_decay * self.w2
        dah = dlogits @ self.w2.T
        dh = adjacency_norm.T @ dah
        dz1 = dh * (z1 > 0)
        dw1 = ax.T @ dz1 + 2 * weight_decay * self.w1

        # 参数更新: W = W - lr * grad。
        self.w1 -= lr * dw1
        self.w2 -= lr * dw2
        return float(loss)

    def predict(self, adjacency_norm, x):
        """返回每个节点预测类别和原始 logits。"""
        logits, _ = self.forward(adjacency_norm, x)
        return logits.argmax(axis=1), logits


def visualize_embeddings(embeddings, labels, filename, title):
    """把二维节点表示画成散点图，不同类别用不同颜色。"""
    plt.figure(figsize=(6, 5))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="Set2", s=35)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)


def main():
    # 1. 构造 Cora 风格图数据，并计算归一化邻接矩阵。
    x, y, adjacency, train_mask, test_mask = make_cora_like_graph()
    adjacency_norm = normalization(adjacency)
    print(f"Number of Node {x.shape[0]}")
    print(f"Number of Edge {adjacency.nnz // 2}")
    print(f"Average Node degree: {adjacency.nnz / x.shape[0]:.2f}")

    # 2. 未训练模型的 logits 前两维可视化，观察随机初始化下的节点分布。
    model = NumpyGCN(num_features=x.shape[1], hidden_channels=16, num_classes=7)
    _, initial_logits = model.predict(adjacency_norm, x)
    visualize_embeddings(initial_logits[:, :2], y, OUTPUT_DIR / "gcn_initial_embeddings.png", "GCN initial node embeddings")

    # 3. 训练 100 个 epoch，并每 50 个 epoch 打印一次损失。
    losses = []
    for epoch in range(1, 101):
        loss = model.train_step(adjacency_norm, x, y, train_mask)
        losses.append(loss)
        if epoch % 50 == 0:
            print(f"Epoch:{epoch:03d}, loss:{loss:.4f}")

    # 4. 在测试节点上计算准确率，并保存训练后的节点表示和损失曲线。
    pred, trained_logits = model.predict(adjacency_norm, x)
    test_acc = accuracy_score(y[test_mask], pred[test_mask])
    print(f"Test Acc:{test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("GCN node classification loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gcn_node_loss.png", dpi=150)
    visualize_embeddings(trained_logits[:, :2], y, OUTPUT_DIR / "gcn_trained_embeddings.png", "GCN trained node embeddings")
    print("图像已保存: output/gcn_initial_embeddings.png, output/gcn_trained_embeddings.png, output/gcn_node_loss.png")


if __name__ == "__main__":
    main()
