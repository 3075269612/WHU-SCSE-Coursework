"""
5.3.2 使用 KNN 进行鸢尾花分类
==============================
对应教材第三次实验 K 最近邻案例，同时包含手写 KNN 核心逻辑和
scikit-learn KNeighborsClassifier 对照实现。
"""

from pathlib import Path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# 引入我们刚才在5.3.1中编写的预处理函数
import sys
import importlib.util
PREPROCESS_FILE = Path(__file__).with_name("5.3.1_数据集的预处理.py")
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_FILE)
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess"] = preprocess_module
spec.loader.exec_module(preprocess_module)
load_and_preprocess_data = preprocess_module.load_and_preprocess_data

# ================================
# 手动编写实现KNN算法的部分核心功能
# ================================

# （1）计算距离：编写一个函数来计算测试样本与所有训练样本之间的距离。
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# （2）找到最近的 K 个邻居：对于每个测试样本，找出最近的 K 个邻居。
def get_neighbors(X_train, test_sample, k):
    distances = [euclidean_distance(test_sample, x) for x in X_train]
    sorted_indices = np.argsort(distances)
    return sorted_indices[:k]

# （3）进行投票：对于分类问题，采用“多数投票”法来决定最终类别。
def predict_classification(X_train, y_train, test_sample, k):
    neighbors_indices = get_neighbors(X_train, test_sample, k)
    neighbors_labels = [y_train[i] for i in neighbors_indices]
    prediction = max(set(neighbors_labels), key=neighbors_labels.count)
    return prediction

def manual_knn_test(X_train, y_train, X_test, y_test):
    # （4）测试算法：测试集评估模型性能。
    y_pred = [predict_classification(X_train, y_train, test, 5) for test in X_test]
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print("手动实现KNN Accuracy:", accuracy)

# ================================
# 使用 Scikit-learn 快速实现
# ================================
def sklearn_knn_test(X, y, X_train_scaled, y_train, X_test_scaled, y_test):
    # （5）创建 KNN 分类器
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    
    # （6）训练模型
    knn.fit(X_train_scaled, y_train)
    
    # （7）模型评估
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Scikit-learn KNN Accuracy: {accuracy:.2f}")
    
    print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("分类报告:\n", classification_report(y_test, y_pred))

    # （8）调整 K 值与交叉验证
    # 交叉验证可以评估模型泛化能力
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    print(f"10折交叉验证平均准确率: {scores.mean():.2f}")

if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()
    
    print("--- 运行手动实现的 KNN ---")
    # 手动实现的KNN没有使用标准化数据以便对照，或者你也可以传入标准化数据
    manual_knn_test(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n--- 运行 Scikit-learn KNN ---")
    sklearn_knn_test(X, y, X_train_scaled, y_train, X_test_scaled, y_test)
