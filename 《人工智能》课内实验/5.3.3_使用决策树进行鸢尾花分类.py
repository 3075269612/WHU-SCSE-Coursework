import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

import sys
import importlib.util
spec = importlib.util.spec_from_file_location("preprocess", "5.3.1_数据集的预处理.py")
preprocess_module = importlib.util.module_from_spec(spec)
sys.modules["preprocess"] = preprocess_module
spec.loader.exec_module(preprocess_module)
load_and_preprocess_data = preprocess_module.load_and_preprocess_data

# ================================
# 手动编写实现决策树的部分核心功能
# ================================

def calculate_entropy(y):
    # （1）计算数据集中标签的熵
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(y, y_left, y_right):
    # （2）计算数据集分裂后的信息增益
    p = len(y_left) / len(y)
    gain = calculate_entropy(y) - p * calculate_entropy(y_left) - (1 - p) * calculate_entropy(y_right)
    return gain

# 2. 寻找最优的划分点
def find_best_split(X, y):
    best_gain = 0
    best_split = None
    n_features = X.shape[1]
    
    for feature_index in range(n_features):
        values = X[:, feature_index]
        unique_values = np.unique(values)
        
        for threshold in unique_values:
            left_indices = values <= threshold
            right_indices = values > threshold
            
            y_left = y[left_indices]
            y_right = y[right_indices]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gain = calculate_information_gain(y, y_left, y_right)
            
            if gain > best_gain:
                best_gain = gain
                best_split = {
                    'feature_index': feature_index,
                    'threshold': threshold,
                    'left_indices': left_indices,
                    'right_indices': right_indices
                }
                
    # 如果没有找到有效的划分，返回 0 作为增益
    if best_split is None:
        return 0, None
    return best_gain, best_split

# 3. 构建递归树节点
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X, y, depth=0, max_depth=3):
    n_samples, n_features = X.shape
    n_labels = len(np.unique(y))
    
    # 停止条件
    if n_labels == 1 or depth >= max_depth:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)
    
    # 找到最佳分裂
    gain, split = find_best_split(X, y)
    
    if gain == 0 or split is None:
        leaf_value = np.bincount(y).argmax()
        return DecisionTreeNode(value=leaf_value)
    
    left_subtree = build_tree(X[split['left_indices']], y[split['left_indices']], depth + 1, max_depth)
    right_subtree = build_tree(X[split['right_indices']], y[split['right_indices']], depth + 1, max_depth)
    
    return DecisionTreeNode(feature_index=split['feature_index'], threshold=split['threshold'], left=left_subtree, right=right_subtree)

# 4. 通过决策树来进行预测
# （1）定义预测函数
def predict(sample, tree):
    if tree.value is not None:
        return tree.value
    
    feature_value = sample[tree.feature_index]
    if feature_value <= tree.threshold:
        return predict(sample, tree.left)
    else:
        return predict(sample, tree.right)

# ================================
# 主程序：包含手动实现和 sklearn 实现
# ================================
if __name__ == "__main__":
    X, y, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess_data()
    
    print("--- 运行手动实现的决策树 ---")
    # （2）在数据集上测试决策树
    tree = build_tree(X_train, y_train, max_depth=3)
    predictions = [predict(sample, tree) for sample in X_test]
    accuracy = np.mean(predictions == y_test)
    print(f"手动决策树分类准确率: {accuracy:.2f}")

    print("\n--- 运行 Scikit-learn 决策树 ---")
    # 5. 使用 Sckit-learn 构建决策树分类器
    # （1）创建决策树实例
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=4)
    # （2）训练模型
    clf.fit(X_train, y_train)
    # （3）模型评估
    y_pred = clf.predict(X_test)
    accuracy_sk = accuracy_score(y_test, y_pred)
    print(f'Accuracy of the model: {accuracy_sk:.2f}')
    
    print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
    print("\n分类报告:\n", classification_report(y_test, y_pred))
    
    print("\n--- 6. 优化决策树模型 ---")
    # ① 预剪枝
    clf_pre = DecisionTreeClassifier(max_depth=5, min_samples_split=4, min_samples_leaf=2)
    clf_pre.fit(X_train, y_train)
    print("预剪枝模型准确率:", accuracy_score(y_test, clf_pre.predict(X_test)))
    
    # ② 后剪枝
    clf_post = DecisionTreeClassifier(ccp_alpha=0.01)
    clf_post.fit(X_train, y_train)
    print("后剪枝模型准确率:", accuracy_score(y_test, clf_post.predict(X_test)))
    
    # （2）交叉验证
    clf_cv = DecisionTreeClassifier(max_depth=5)
    scores = cross_val_score(clf_cv, X, y, cv=5)
    print("交叉验证准确率: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
