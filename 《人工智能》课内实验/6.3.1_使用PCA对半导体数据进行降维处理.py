"""
6.3.1 使用 PCA 对半导体制造数据进行降维处理
================================================
问题背景：
  半导体制造过程中有 591 个传感器采集生产数据，但如此高维的数据难以
  可视化和分析。许多传感器之间存在相关性，数据包含大量冗余。
  PCA（主成分分析）通过线性变换将数据投影到低维空间，同时保留最大
  方差——即保留数据中最重要的"变化模式"。

算法直觉：
  想象你在不同角度拍摄一个 3D 物体，然后选出 2 张最能展现物体全貌的
  照片。PCA 做的事类似：找到数据变化最大的方向（主成分），用这些方向
  作为新坐标轴，丢掉变化小的方向，实现降维。

数据说明：
  secom.data — UCI 半导体制造数据集，1567 个样本 × 591 个传感器读数，
  部分传感器存在故障导致的缺失值（NaN）。
"""

import numpy as np
from numpy import *
mat = np.matrix  # NumPy 2.x 兼容：mat 别名已被移除
import matplotlib.pyplot as plt

# ================================
# 1. 数据加载与预处理
# ================================

def loadDataSet(fileName, delim='\t'):
    """
    通用数据加载函数：从文本文件读取数据并转换为矩阵
    参数：
      fileName : 文件路径
      delim    : 列分隔符，默认为制表符 '\t'
    返回：
      numpy.matrix 格式的数据矩阵
    """
    fr = open(fileName)
    # 按行读取，去除首尾空白，按分隔符切分
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # 将字符串转换为浮点数
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def replaceNanWithMean():
    """
    缺失值处理：用该特征的均值替换 NaN

    半导体制造中，传感器故障会导致某些读数缺失（NaN）。
    简单而有效的策略是用该传感器的平均读数填充——不删除整行，
    因为其他传感器的数据仍然有价值。

    步骤：
      （1）遍历每个特征列
      （2）计算该列中非 NaN 值的均值
      （3）将该列的 NaN 替换为该均值
    """
    # 加载 secom.data，使用空格作为分隔符
    datMat = loadDataSet('data/secom.data', ' ')
    numFeat = shape(datMat)[1]  # 获取特征数量（列数）

    for i in range(numFeat):
        # 找出第 i 列中所有非 NaN 值的索引
        # datMat[:, i].A 将矩阵第 i 列转为数组，isnan 判断是否 NaN
        valid_indices = nonzero(~isnan(datMat[:, i].A))[0]

        # 计算非 NaN 值的均值
        meanVal = mean(datMat[valid_indices, i])

        # 找出 NaN 的索引，用均值替换
        nan_indices = nonzero(isnan(datMat[:, i].A))[0]
        datMat[nan_indices, i] = meanVal

    return datMat


# ================================
# 2. PCA 主成分分析
# ================================

def analyse_data(dataMat):
    """
    分析并打印各主成分的方差贡献率

    PCA 中每个主成分对应一个特征值。特征值越大，说明该方向上的
    数据变化越大，包含的信息越多。

    方差占比 = 该特征值 / 所有特征值之和
    累积方差占比 = 前 N 个特征值之和 / 所有特征值之和

    累积方差占比越高，说明降维后的数据保留的信息越多。
    通常我们关注累积占比达到 85%~95% 时需要多少个主成分。
    """
    # （1）去中心化：每列减去该列的均值，使数据中心移到原点
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # （2）计算协方差矩阵
    #     协方差矩阵描述各特征之间的线性相关关系
    #     C[i][j] 表示特征 i 和特征 j 的协方差
    covMat = cov(meanRemoved, rowvar=0)  # rowvar=0：每列是一个变量

    # （3）对协方差矩阵做特征值分解
    #     特征值 = 该方向上的方差大小
    #     特征向量 = 主成分方向
    eigvals, eigVects = linalg.eig(mat(covMat))

    # （4）将特征值从大到小排序
    eigValInd = argsort(eigvals)  # 升序排列的索引

    # 选取前 topNfeat 个主成分
    topNfeat = 20
    # 取最后 21 个索引，步长 -1 实现倒序（从大到小）
    eigValInd = eigValInd[:-(topNfeat + 1):-1]

    # （5）计算并打印各主成分的方差占比
    cov_all_score = float(sum(eigvals))  # 所有特征值之和
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print(" 主成分：{}, 方差占比： {:.2f}%, 累积方差占比： {:.2f}%".format(
            i + 1,
            line_cov_score / cov_all_score * 100,
            sum_cov_score / cov_all_score * 100
        ))


# ================================
# 3. 数据重构与可视化
# ================================

def show_picture(dataMat, reconMat):
    """
    降维-重构对比散点图

    取原始数据和重构数据的前两个维度做散点图：
      - 蓝色三角 (^) : 原始数据的前两维
      - 红色圆圈 (o) : 降维后重构数据的前两维

    两者越接近，说明降维过程丢失的信息越少。
    如果圆圈和三角大面积重叠，说明 PCA 保留效果好。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # 原始数据散点：蓝色三角
    ax.scatter(dataMat[:, 0].flatten().A[0],
               dataMat[:, 1].flatten().A[0],
               marker='^', s=90)

    # 重构数据散点：红色圆圈
    ax.scatter(reconMat[:, 0].flatten().A[0],
               reconMat[:, 1].flatten().A[0],
               marker='o', s=50, c='red')

    plt.show()


# ================================
# 主程序：完整 PCA 降维流程
# ================================
if __name__ == '__main__':
    try:
        # ---- 步骤1：加载数据并处理缺失值 ----
        dataMat = replaceNanWithMean()

        # ---- 步骤2：去中心化 ----
        # 使每个特征的均值为 0，这是 PCA 的前提条件
        meanVals = mean(dataMat, axis=0)
        meanRemoved = dataMat - meanVals

        # ---- 步骤3：计算协方差矩阵 ----
        # 协方差矩阵是 PCA 的核心：它衡量各特征之间的相关性
        covMat = cov(meanRemoved, rowvar=0)

        # ---- 步骤4：特征值分解 ----
        # 求协方差矩阵的特征值和特征向量
        # 每个特征值 = 对应主成分方向上的方差
        # 每个特征向量 = 一个主成分的方向
        eigVals, eigVects = linalg.eig(mat(covMat))

        # ---- 步骤5：选取前 20 个主成分 ----
        eigValInd = argsort(eigVals)  # 特征值升序排列
        topNfeat = 20
        eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 取最大的 20 个

        # 选出对应的特征向量作为投影矩阵
        redEigVects = eigVects[:, eigValInd]  # 591 × 20 的投影矩阵

        # ---- 步骤6：降维 ----
        # 将 1567 × 591 的数据投影到 20 维空间
        # lowDDataMat: 1567 × 20（降维后的低维表示）
        lowDDataMat = meanRemoved * redEigVects

        # ---- 步骤7：从低维重构原始数据 ----
        # 用低维表示和投影矩阵反推回原始空间
        # 重构 = 低维表示 × (投影矩阵)^T + 原始均值
        # 这一步会丢失信息：因为被丢弃的 571 个主成分无法恢复
        reconMat = (lowDDataMat * redEigVects.T) + meanVals

        # ---- 步骤8：分析和可视化 ----
        analyse_data(dataMat)         # 打印方差贡献率
        show_picture(dataMat, reconMat)  # 原始 vs 重构对比图

    except Exception as e:
        print("运行出错，通常是因为缺失数据集 (secom.data):", e)
