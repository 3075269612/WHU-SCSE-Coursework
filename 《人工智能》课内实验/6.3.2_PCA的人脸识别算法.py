"""
6.3.2 基于 PCA 的人脸识别算法（特征脸 / Eigenface）
=====================================================
问题背景：
  人脸图像的原始维度极高。ORL 数据集中每张人脸是 92×112 = 10,304 维。
  直接在如此高维的空间做分类会面临"维数灾难"——计算量巨大且模型泛化
  能力差。PCA 可以将 10,304 维压缩到几十维，用低维特征做识别。

算法流程：
  1. 加载 ORL 人脸数据集（40 人 × 10 张 = 400 张照片）
  2. 每人随机选 k 张作为训练集，其余作为测试集
  3. 用 PCA 将训练人脸降到 r 维（默认为 10 维）
  4. 对每张测试人脸，在低维空间找最近的训练人脸
  5. 最近邻居的类别即为预测结果

关键技巧：
  当样本数 N（如 200）远小于特征数 D（10,304）时，直接计算 D×D
  的协方差矩阵并做特征分解是不现实的。这里使用简化技巧：
    先计算 N×N 的小矩阵 C = A·A^T 的特征向量，
    再通过 V = A^T·V' 映射回原始 10,304 维空间。
  这正是"特征脸（Eigenface）"算法的数学基础。

数据说明：
  ORL/AT&T 人脸数据库，40 人，每人 10 张 92×112 的灰度照片。
  拍摄于不同时间，包含表情、光照、是否戴眼镜等变化。
"""

import cv2
import numpy as np


# ================================
# 1. 图像预处理
# ================================

def img2vector(image):
    """
    将图像文件读取为 1 维向量

    输入：图像文件路径
    输出：1 × (行×列) 的一维行向量

    对于 92×112 的图像，输出是 1×10304 的向量。
    这样每张人脸就变成了高维空间中的一个点。
    """
    # cv2.imread(image, 0): 以灰度模式读取图像
    img = cv2.imread(image, 0)
    if img is None:
        raise ValueError(f"无法读取图片 {image}")

    rows, cols = img.shape
    # 将 2D 图像重塑为 1D 向量
    imgVector = np.reshape(img, (1, rows * cols))
    return imgVector


# ================================
# 2. 加载 ORL 人脸数据集
# ================================

def load_orl(k):
    """
    加载 ORL 人脸数据集，按每人 k 张训练、(10-k) 张测试划分

    参数：
      k : 每人用于训练的图片数量（1 ≤ k ≤ 9）

    返回：
      train_face : (40*k) × 10304 的训练人脸矩阵
      train_label: (40*k) 维训练标签（1~40 表示人物编号）
      test_face  : (40*(10-k)) × 10304 的测试人脸矩阵
      test_label : (40*(10-k)) 维测试标签

    数据集结构：
      data/ORL/s{1-40}/{1-10}.jpg
      其中 sX 表示第 X 个人，Y.jpg 是该人的第 Y 张照片
    """
    # 预分配内存：每行是一张人脸（10304 维），训练集 + 测试集
    train_face = np.zeros((40 * k, 112 * 92))
    train_label = np.zeros(40 * k)
    test_face = np.zeros((40 * (10 - k), 112 * 92))
    test_label = np.zeros(40 * (10 - k))

    orlpath = 'data/ORL'

    # 随机打乱 1-10 的排列，保证每次运行时训练/测试划分不同
    sample = np.random.permutation(10) + 1  # 输出 [1..10] 的随机排列

    for i in range(40):             # 遍历 40 个人
        people_num = i + 1          # 人物编号 1~40
        for j in range(10):         # 遍历该人的 10 张照片
            # 构建图片路径，如: data/ORL/s1/3.jpg
            image = orlpath + '/s' + str(people_num) + '/' + str(sample[j]) + '.jpg'

            # 读取图片并转为 1×10304 向量
            img = img2vector(image)

            if j < k:
                # 前 k 张 → 训练集
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                # 后 (10-k) 张 → 测试集
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label


# ================================
# 3. PCA 降维（特征脸算法）
# ================================

def PCA(data, r):
    """
    主成分分析降维

    参数：
      data : 训练数据矩阵，每行一个样本
      r    : 降维后的维度（保留的主成分数量）

    返回：
      final_data : 降维后的数据矩阵
      data_mean  : 原始数据的均值（用于后续归一化测试数据）
      V_r        : 投影矩阵（主成分方向）

    算法细节（样本数 << 特征数时的优化）：
      传统 PCA: 对 D×D 协方差矩阵做分解（D=10304 时不可能）
      优化 PCA: 计算 N×N 小矩阵 C = A·A^T 的特征向量（N=200 很轻松），
               再利用 V = A^T·V_small 映射回原始空间。
               这一技巧的对偶关系是：
                 A·A^T 的特征向量 v_small  →  A^T·A 的特征向量 A^T·v_small
    """
    # （1）数据准备
    data = np.float32(np.matrix(data))
    rows, cols = np.shape(data)        # rows=样本数, cols=10304

    # （2）去中心化：每列减去均值
    data_mean = np.mean(data, 0)       # 1×10304 的均值向量
    A = data - np.tile(data_mean, (rows, 1))  # 每行减去均值（广播）

    # （3）计算 N×N 的小协方差矩阵（关键优化！）
    #     直接算 10304×10304 的协方差矩阵太慢，现在只算 200×200
    C = A * A.T                       # C 是 N×N 矩阵

    # （4）对小矩阵做特征值分解
    D, V = np.linalg.eig(C)            # D: 特征值, V: N×N 特征向量矩阵

    # （5）取前 r 个最大特征值对应的特征向量（r=10）
    V_r = V[:, 0:r]

    # （6）映射回原始空间：V_original = A^T · V_small
    #     现在 V_r 中的每一列都是原始 10304 维空间中的主成分方向
    V_r = A.T * V_r

    # （7）归一化：将每个主成分向量变为单位向量
    for i in range(r):
        V_r[:, i] = V_r[:, i] / np.linalg.norm(V_r[:, i])

    # （8）将原始数据投影到低维空间
    final_data = A * V_r               # N×r 的低维表示

    return final_data, data_mean, V_r


# ================================
# 4. 人脸识别主流程
# ================================

def face_recongize():
    """
    完整的人脸识别流程：

    1. 加载数据（每人 5 张训练，5 张测试）
    2. PCA 降维到 10 维
    3. 对每张测试脸，在低维空间找最近训练脸（最近邻分类）
    4. 计算准确率
    """
    try:
        # ---- 步骤1：加载数据集 ----
        # 每人随机选 5 张训练，剩余 5 张测试
        # 训练集：200 张，测试集：200 张
        train_face, train_label, test_face, test_label = load_orl(5)

        # 用于记录不同参数组合的准确率
        x_value = []
        y_value = []

        # ---- 步骤2：PCA 降维 ----
        # 将 200 张 10304 维的训练脸降到 10 维
        # data_train_new : 200 × 10
        # data_mean       : 1 × 10304（训练集的均值脸）
        # V_r             : 10304 × 10（10 个特征脸，每个是 10304 维）
        data_train_new, data_mean, V_r = PCA(train_face, 10)

        num_train = data_train_new.shape[0]  # 200
        num_test = test_face.shape[0]         # 200

        # ---- 步骤3：将测试脸投影到同一个 PCA 空间 ----
        # 注意：测试脸必须用训练集的均值去中心化，
        #      然后用同一个投影矩阵 V_r 做降维！
        temp_face = test_face - np.tile(data_mean, (num_test, 1))
        data_test_new = temp_face * V_r       # 200 × 10

        data_test_new = np.array(data_test_new)
        data_train_new = np.array(data_train_new)

        # ---- 步骤4：最近邻分类 ----
        # 对每张测试脸，在低维空间找到距离最近的训练脸，
        # 该训练脸的人物编号即为预测结果
        true_num = 0  # 正确预测的计数
        for i in range(num_test):
            testFace = data_test_new[i, :]      # 1 × 10 的测试脸

            # 计算与所有训练脸的欧氏距离
            diffMat = data_train_new - np.tile(testFace, (num_train, 1))
            sqDiffMat = diffMat ** 2             # 逐元素平方
            sqDistances = sqDiffMat.sum(axis=1)  # 每行求和 = 距离²

            # 排序，取最近的训练脸索引
            sortedDistIndicies = sqDistances.argsort()
            indexMin = sortedDistIndicies[0]      # 最近邻居

            # 比较最近邻居的标签和真实标签
            if train_label[indexMin] == test_label[i]:
                true_num += 1

        # ---- 步骤5：计算并输出准确率 ----
        accuracy = float(true_num) / num_test
        x_value.append(5)
        y_value.append(round(accuracy, 2))

        print('当对每个人随机选择 %d 张照片降低至 %d 维进行训练时，The classify accuracy is: %.2f%%'
              % (5, 10, accuracy * 100))

    except Exception as e:
        print("运行出错，通常是因为缺失 ORL 数据集:", e)


# ================================
# 主程序入口
# ================================
if __name__ == '__main__':
    face_recongize()
