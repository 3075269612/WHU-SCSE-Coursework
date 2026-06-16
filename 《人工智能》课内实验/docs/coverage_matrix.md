# 实验 PDF 覆盖矩阵

本矩阵按 `docs/` 中 5 份 PDF 核对案例章节、代码位置、数据依赖和覆盖状态。脚本默认从项目根目录运行。

## 第一次实验内容1.pdf

| PDF 章节 | 案例 | 对应代码 | 数据依赖 | 状态 |
| --- | --- | --- | --- | --- |
| 2.3.1 | 八数码问题 | `experiments/01_search/2.3.1_八数码问题.py` | 无 | 已覆盖 |
| 2.3.2 | 传教士与野人问题 | `experiments/01_search/2.3.2_传教士与野人问题.py` | 无 | 已覆盖 |
| 2.3.3 | 猴子和香蕉问题 | `experiments/01_search/2.3.3_猴子和香蕉问题.py` | 无 | 已覆盖 |
| 2.3.4 | 汉诺塔问题 | `experiments/01_search/2.3.4_汉诺塔问题.py` | 无 | 已覆盖 |
| 2.3.5 | 0-1 背包问题 | `experiments/01_search/2.3.5_01背包问题.py` | 无 | 已覆盖 |
| 2.3.6 | 旅行商问题 | `experiments/01_search/2.3.6_旅行商问题.py` | 无 | 已覆盖 |

## 第二次实验内容.pdf

| PDF 章节 | 案例 | 对应代码 | 数据依赖 | 状态 |
| --- | --- | --- | --- | --- |
| 3.3.1 | 命题逻辑案例 | `experiments/02_logic_and_optimization/3.3.1_三老师授课分配问题.py` | 无 | 已覆盖 |
| 3.3.2 | 谓词逻辑案例 | `experiments/02_logic_and_optimization/3.3.2_图书管理系统问题.py` | 无 | 已覆盖 |
| 3.3.3 | 归结推理案例 | `experiments/02_logic_and_optimization/3.3.3_小华带电脑问题.py` | `data/S.txt`，缺失时自动生成 | 已覆盖 |
| 4.3.1 | 遗传算法 TSP | `experiments/02_logic_and_optimization/4.3.1_遗传算法求解TSP问题.py` | 无 | 已覆盖 |
| 4.3.2 | 蚁群算法 TSP | `experiments/02_logic_and_optimization/4.3.2_蚁群算法求解TSP问题.py` | 无 | 已覆盖 |
| 4.3.3 | 模拟退火算法 TSP | `experiments/02_logic_and_optimization/4.3.3_模拟退火算法求解TSP问题.py` | 无 | 已覆盖 |

## 第三次实验内容.pdf

| PDF 章节 | 案例 | 对应代码 | 数据依赖 | 状态 |
| --- | --- | --- | --- | --- |
| 5.3.1 | 数据集预处理 | `experiments/03_machine_learning/5.3.1_数据集的预处理.py` | scikit-learn Iris | 已覆盖 |
| 5.3.2 | KNN 鸢尾花分类 | `experiments/03_machine_learning/5.3.2_使用KNN进行鸢尾花分类.py` | scikit-learn Iris | 已覆盖 |
| 5.3.3 | 决策树鸢尾花分类 | `experiments/03_machine_learning/5.3.3_使用决策树进行鸢尾花分类.py` | scikit-learn Iris | 已覆盖 |
| 5.3.4 | 朴素贝叶斯鸢尾花分类 | `experiments/03_machine_learning/5.3.4_使用朴素贝叶斯进行鸢尾花分类.py` | scikit-learn Iris | 已覆盖 |
| 5.3.5 | SVM 鸢尾花分类 | `experiments/03_machine_learning/5.3.5_使用支持向量机进行鸢尾花分类.py` | scikit-learn Iris | 已覆盖 |
| 6.3.1 | PCA 半导体数据降维 | `experiments/03_machine_learning/6.3.1_使用PCA对半导体数据进行降维处理.py` | `data/secom.data` | 已覆盖 |
| 6.3.2 | PCA 人脸识别 | `experiments/03_machine_learning/6.3.2_PCA的人脸识别算法.py` | `data/ORL/` | 已覆盖 |
| 6.3.3 | K-means 聚类分析 | `experiments/03_machine_learning/6.3.3_使用K-means对数据集进行聚类分析.py` | scikit-learn 生成数据、`data/testSet.txt` | 已覆盖 |

## 第四次实验内容.pdf

| PDF 章节 | 案例 | 对应代码 | 数据依赖 | 状态 |
| --- | --- | --- | --- | --- |
| 7.3.1 | 数据处理 / 激活函数可视化 | `experiments/04_neural_networks/7.3.1_激活函数可视化.py` | PyTorch | 已覆盖，需可选深度学习依赖 |
| 7.3.2 | 模型搭建 | `experiments/04_neural_networks/7.3.2_MNIST手写数字识别.py` | `data/MNIST/`、PyTorch | 已覆盖，需可选深度学习依赖 |
| 7.3.3 | 模型训练 | `experiments/04_neural_networks/7.3.2_MNIST手写数字识别.py` | `data/MNIST/`、PyTorch | 已覆盖，单脚本覆盖 |
| 7.3.4 | 模型评估 | `experiments/04_neural_networks/7.3.2_MNIST手写数字识别.py` | `data/MNIST/`、PyTorch | 已覆盖，单脚本覆盖 |
| 8.3.1 | 准备数据 / CNN 基础组件 | `experiments/04_neural_networks/8.3.1_CNN基础组件实现.py`、`experiments/04_neural_networks/8.3.2_FashionMNIST时装分类.py` | NumPy、`data/FashionMNIST/` | 已覆盖 |
| 8.3.2 | 搭建模型 | `experiments/04_neural_networks/8.3.2_FashionMNIST时装分类.py` | `data/FashionMNIST/`、PyTorch | 已覆盖，需可选深度学习依赖 |
| 8.3.3 | 训练模型 | `experiments/04_neural_networks/8.3.2_FashionMNIST时装分类.py` | `data/FashionMNIST/`、PyTorch | 已覆盖，单脚本覆盖 |
| 8.3.4 | 测试模型 | `experiments/04_neural_networks/8.3.2_FashionMNIST时装分类.py` | `data/FashionMNIST/`、PyTorch | 已覆盖，单脚本覆盖 |

## 第六次实验内容(1).pdf

| PDF 章节 | 案例 | 对应代码 | 数据依赖 | 状态 |
| --- | --- | --- | --- | --- |
| 9.3.1 | RNN 姓氏分类 | `experiments/05_sequence_and_graph/9.3.1_RNN模型案例_姓氏分类.py` | `data/names/*.txt` 可选；缺失时用内置小数据 | 已覆盖，需可选深度学习依赖 |
| 9.3.2 | LSTM 影评情感分析 | `experiments/05_sequence_and_graph/9.3.2_LSTM模型案例_影评情感分析.py` | 内置小型影评数据 | 已覆盖；用 NumPy 复现流程 |
| 9.3.3 | Transformer 英法机器翻译 | `experiments/05_sequence_and_graph/9.3.3_Transformer模型案例_英法机器翻译.py` | `fra-eng/fra.txt` 可选；缺失时用内置小语料 | 已覆盖，需可选深度学习依赖 |
| 10.3 | GCN 节点分类 / Cora 案例 | `experiments/05_sequence_and_graph/10.3.1_基于GCN的节点分类.py` | 离线 Cora 风格小图 | 已覆盖；替代官方 Cora/PyG 数据流 |

## 审查结论

- 五份 PDF 中课堂案例代码均有对应脚本。
- 需要官方在线数据的案例均提供离线回退或本地数据说明。
- 深度学习相关脚本依赖 PyTorch/PyTorchVision，已放入 `pyproject.toml` 的 `deep-learning` 可选依赖组。
- 代码路径已改为基于项目根目录解析，移动到章节目录后仍可从项目根目录运行。
