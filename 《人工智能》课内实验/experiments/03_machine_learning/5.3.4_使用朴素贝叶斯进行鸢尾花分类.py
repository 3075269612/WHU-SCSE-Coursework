"""
5.3.4 使用朴素贝叶斯进行鸢尾花分类
==================================
对应教材第三次实验朴素贝叶斯案例，包含高斯朴素贝叶斯手写预测流程和
scikit-learn GaussianNB 对照实现。
"""

from pathlib import Path
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import sys
import importlib.util
PREPROCESS_FILE = Path(__file__).with_name("5.3.1_数据集的预处理.py")
spec = importlib.util.spec_from_file_location("preprocess", PREPROCESS_FILE)
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess"] = preprocess_module
spec.loader.exec_module(preprocess_module)
load_and_preprocess_data = preprocess_module.load_and_preprocess_data

# ================================
# 手动编写实现朴素贝叶斯算法的核心功能
# ================================

# （1）先验概率：计算训练集中每个类别的占比
def calculate_prior(y):
    classes = np.unique(y)
    prior_probabilities = {}
    for cls in classes:
        prior_probabilities[cls] = np.sum(y == cls) / len(y)
    return prior_probabilities

# （2）条件概率：对于连续特征，朴素贝叶斯假设特征符合正态分布，
# 因此可根据各类别特征的均值和方差计算对应的条件概率
def calculate_mean_variance(X, y):
    classes = np.unique(y)
    mean_variance = {}
    for cls in classes:
        X_class = X[y == cls]
        mean_variance[cls] = {
            "mean": np.mean(X_class, axis=0),
            "var": np.var(X_class, axis=0)
        }
    return mean_variance

# 计算概率密度函数
def gaussian_probability(x, mean, var):
    eps = 1e-6  # 防止方差为零的数值问题
    coefficient = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponent = np.exp(-((x - mean) ** 2) / (2.0 * var + eps))
    return coefficient * exponent

# （3）后验概率：根据贝叶斯定理进行预测
def predict_naive_bayes(X, priors, mean_variance_params):
    predictions = []
    classes = list(priors.keys())
    for i in range(X.shape[0]):
        posteriors = {}
        for cls in classes:
            prior = np.log(priors[cls])  # 取对数防止浮点数下溢
            conditional = 0
            for j in range(X.shape[1]):
                mean = mean_variance_params[cls]["mean"][j]
                var = mean_variance_params[cls]["var"][j]
                conditional += np.log(gaussian_probability(X[i, j], mean, var))
            posteriors[cls] = prior + conditional
        predictions.append(max(posteriors, key=posteriors.get))
    return np.array(predictions)

# ================================
# 主程序：包含手动实现和 sklearn 实现
# ================================
if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()
    
    print("--- 运行手动实现的朴素贝叶斯 ---")
    priors = calculate_prior(y_train)
    print("先验概率:", priors)
    
    mean_variance_params = calculate_mean_variance(X_train_scaled, y_train)
    # print("均值和方差:", mean_variance_params) # 取消注释可查看具体均值与方差
    
    y_pred_manual = predict_naive_bayes(X_test_scaled, priors, mean_variance_params)
    accuracy_manual = accuracy_score(y_test, y_pred_manual)
    print(f"手动朴素贝叶斯准确率: {accuracy_manual:.2f}")

    print("\n--- 运行 Scikit-learn 朴素贝叶斯分类器 ---")
    # 朴素贝叶斯模型训练
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = nb_model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the model: {accuracy:.2f}')
    print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("\n分类报告:\n", classification_report(y_test, y_pred))
