# 人工智能课内实验代码复现

本项目整理了《人工智能》课内 5 次实验 PDF 中要求复现的案例代码，覆盖搜索算法、逻辑推理、计算智能、机器学习、神经网络、序列模型和图神经网络等内容。

## 目录结构

```text
.
├── docs/                         # 老师发放的实验 PDF 与覆盖矩阵
├── experiments/
│   ├── 01_search/                # 2.3.x 搜索算法案例
│   ├── 02_logic_and_optimization/# 3.3.x 逻辑推理、4.3.x 计算智能
│   ├── 03_machine_learning/      # 5.3.x 监督学习、6.3.x 无监督学习
│   ├── 04_neural_networks/       # 7.3.x 神经网络、8.3.x CNN
│   └── 05_sequence_and_graph/    # 9.3.x 序列模型、10.3.x 图神经网络
├── data/                         # 本地数据集
├── output/                       # 运行生成的图片、模型等结果
└── tools/                        # 数据准备辅助脚本
```

详细覆盖情况见 [docs/coverage_matrix.md](docs/coverage_matrix.md)。

## 环境准备

本项目优先使用 `uv` 管理 Python 环境：

```bash
uv sync
```

运行脚本时统一从项目根目录执行，例如：

```bash
uv run python experiments/01_search/2.3.1_八数码问题.py
uv run python experiments/03_machine_learning/5.3.2_使用KNN进行鸢尾花分类.py
```

深度学习案例需要额外安装 PyTorch：

```bash
uv sync --extra deep-learning
```

如果所在平台安装 PyTorch 需要指定 CPU/CUDA 源，请按 PyTorch 官网命令安装后再运行对应脚本。

## 数据策略

项目默认支持离线验收：

- `data/secom.data`、`data/testSet.txt`、`data/ORL/`、`data/MNIST/`、`data/FashionMNIST/` 已按当前课程目录保留。
- `9.3.1` 姓氏分类若没有 `data/names/*.txt`，会使用内置小型姓氏数据。
- `9.3.2` 情感分析使用内置小型影评数据，复现文本向量化、LSTM 编码、二分类训练流程。
- `9.3.3` Transformer 翻译若没有 `fra-eng/fra.txt`，会使用内置小型英法平行语料。
- `10.3.1` GCN 节点分类使用 Cora 风格小图复现节点分类流程，避免依赖在线下载和 `torch_geometric`。

这些替代数据用于保证离线可运行；覆盖矩阵中已标明与教材官方数据集的差异。

## 常用运行命令

轻量案例：

```bash
uv run python experiments/02_logic_and_optimization/3.3.3_小华带电脑问题.py
uv run python experiments/03_machine_learning/6.3.3_使用K-means对数据集进行聚类分析.py
uv run python experiments/05_sequence_and_graph/10.3.1_基于GCN的节点分类.py
```

深度学习案例：

```bash
uv run python experiments/04_neural_networks/7.3.1_激活函数可视化.py
uv run python experiments/04_neural_networks/7.3.2_MNIST手写数字识别.py
uv run python experiments/04_neural_networks/8.3.2_FashionMNIST时装分类.py
uv run python experiments/05_sequence_and_graph/9.3.1_RNN模型案例_姓氏分类.py
```

生成的图片和模型文件会写入 `output/`。

## 说明

- 代码路径均基于项目根目录动态解析，不依赖本机绝对路径。
- 图形脚本默认保存图片，不弹出交互窗口，便于在无 GUI 环境验收。
- `.venv/`、缓存和可再生成输出已加入忽略规则，复制到 Git 仓库后不会污染版本历史。
