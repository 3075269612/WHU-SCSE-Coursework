"""
5.3.5 使用支持向量机进行鸢尾花分类
==================================
对应教材第三次实验支持向量机案例，包含简化手写 SVM、决策边界可视化、
多分类 SVC 以及超参数搜索示例。
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import sys
import importlib.util
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PREPROCESS_FILE = Path(__file__).with_name("5.3.1_数据集的预处理.py")
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_FILE)
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess"] = preprocess_module
spec.loader.exec_module(preprocess_module)
load_and_preprocess_data = preprocess_module.load_and_preprocess_data

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# ================================
# 手动编写实现支持向量机
# ================================

# 1. 核心核函数
# 线性核函数
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# RBF 核函数
def rbf_kernel(x1, x2, gamma=0.5):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# 2. 实现支持向量机算法类
class SVM:
    def __init__(self, kernel=linear_kernel, learning_rate=0.001, lambda_param=0.01, n_iters=100):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.X = X
        self.y = y
        # 梯度下降优化
        for _ in range(self.n_iters):
            for i in range(n_samples):
                condition = y[i] * (self._decision_function(X[i])) < 1
                if condition:
                    self.alpha[i] += self.learning_rate * (1 - y[i] * self._decision_function(X[i]))
        # 更新偏置b
        support_vectors_idx = np.where(self.alpha > 1e-5)[0]
        self.b = np.mean([y[i] - self._decision_function(X[i]) for i in support_vectors_idx]) if len(support_vectors_idx) > 0 else 0

    def _decision_function(self, x):
        return np.sum([self.alpha[i] * self.y[i] * self.kernel(self.X[i], x) for i in range(len(self.alpha))]) - self.b

    def predict(self, X):
        return np.sign([self._decision_function(x) for x in X])

# 4. 可视化决策边界
def plot_decision_boundary(X, y, model, title):
    """绘制并保存二分类决策边界。"""
    x0, x1 = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
    )
    X_grid = np.array([x0.ravel(), x1.ravel()]).T
    y_pred = model.predict(X_grid).reshape(x0.shape)
    
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])
    
    plt.contourf(x0, x1, y_pred, alpha=0.3, cmap=cmap_background)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='k', marker='o', s=50)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title(title)
    safe_name = title.replace(" ", "_").replace("/", "_")
    plt.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()

# ================================
# 主程序
# ================================
if __name__ == "__main__":
    X_full, y_full, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()
    
    # 注意：书中手动编写的简单SVM主要用于二分类问题，并且特征二维时才能画图
    # 这里我们提取只有两个类别以及前两个特征的数据作演示
    idx = y_full != 2
    X_bin = X_full[idx][:, :2]  # 为了画图，仅取前两个特征
    y_bin = y_full[idx]
    y_bin = np.where(y_bin == 0, -1, 1)  # 手动 SVM 中类别多映射为 -1, 1
    
    from sklearn.model_selection import train_test_split
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

    print("--- 运行手动实现的 SVM (二分类降维版用于可视化) ---")
    # 使用线性核初始化并训练模型
    svm_linear = SVM(kernel=linear_kernel, learning_rate=0.001, lambda_param=0.01, n_iters=100)
    svm_linear.fit(X_b_train, y_b_train)

    # 使用RBF 核初始化并训练模型
    svm_rbf = SVM(kernel=rbf_kernel, learning_rate=0.001, lambda_param=0.01, n_iters=100)
    svm_rbf.fit(X_b_train, y_b_train)

    # 预测并评估模型（线性核）
    predictions_linear = svm_linear.predict(X_b_test)
    accuracy_linear = np.mean(predictions_linear == y_b_test)
    print(f"线性核模型的测试集准确率: {accuracy_linear:.2f}")

    # 预测并评估模型（RBF 核）
    predictions_rbf = svm_rbf.predict(X_b_test)
    accuracy_rbf = np.mean(predictions_rbf == y_b_test)
    print(f"RBF核模型的测试集准确率: {accuracy_rbf:.2f}")

    # 绘制并保存决策边界
    print("绘制决策边界并保存到 output/ 目录...")
    plot_decision_boundary(X_bin, y_bin, svm_linear, 'SVM 决策边界（线性核）')
    plot_decision_boundary(X_bin, y_bin, svm_rbf, 'SVM 决策边界（RBF 核）')

    print("\n--- 5. 使用 Scikit-learn 快速实现 SVM 分类器 ---")
    # 创建支持向量机模型
    svm_model = SVC(kernel='linear', C=1.0, random_state=34)
    # 训练模型
    svm_model.fit(X_train_scaled, y_train)
    # 模型评估
    y_pred = svm_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    print("\n--- 6. 参数调优：网格搜索和随机搜索 ---")
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001]
    }
    svm_tune = SVC(random_state=34)

    # 网格搜索
    grid_search = GridSearchCV(estimator=svm_tune, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    print("GridSearchCV Best parameters:", grid_search.best_params_)

    # 随机搜索
    random_search = RandomizedSearchCV(SVC(random_state=34), param_grid, n_iter=10, verbose=0, cv=5, random_state=34)
    random_search.fit(X_train_scaled, y_train)
    print("RandomizedSearchCV Best parameters:", random_search.best_params_)
