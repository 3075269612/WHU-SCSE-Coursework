"""
5.3.1 数据集的预处理
====================
对应教材第三次实验中鸢尾花分类案例的数据准备部分。
本脚本使用 scikit-learn 内置 Iris 数据集，完成数据加载、训练/测试集划分
和特征标准化，供 5.3.2-5.3.5 的分类模型复用。
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    """
    5.3.1 数据集的预处理
    在应用各种模型之前，首先需要对鸢尾花数据集进行预处理。
    预处理包括数据的加载、特征缩放和特征选择。
    """
    # （1）加载数据集：加载用于模型学习的鸢尾花数据集。
    # Scikit-learn库已经继承了这个数据集，直接导入即可：
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # （2）数据分割：为了评估模型的性能，需要将数据集分成训练集和测试集。
    # 通常，使用70%的数据作为训练集，剩余的30%作为测试集：
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    # （3）数据标准化：由于大多数算法（尤其是依赖距离的KNN）不同的特征尺度可能会影响算法性能。
    # 标准化将特征的均值调整为0，标准差调整为1，使各特征具有相同的尺度。
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 返回原始和标准化后的数据，供后续模块使用
    return X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()
    print("数据集预处理完成！")
    print("训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)
