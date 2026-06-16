"""
6.3.3 使用 K-means 对数据集进行聚类分析
============================================
问题背景：
  聚类是无监督学习的代表任务——数据没有标签，我们希望通过算法自动
  发现数据内部的分组结构。K-means 是最经典、最广泛使用的聚类算法。

算法直觉：
  想象你要在教室里摆 K 张桌子，每个学生去离自己最近的桌子。
  等所有人就座后，把桌子挪到该组学生的中心位置。然后学生重新选择
  最近的桌子……重复这个过程直到桌子不再移动。最终每张桌子周围
  的学生形成了一个"簇"。

K-means 核心步骤：
  1. 随机初始化 K 个聚类中心
  2. 分配：将每个样本分配到最近的中心
  3. 更新：重新计算每个簇的中心（均值）
  4. 重复 2-3 直到中心不再变化（或达到最大迭代次数）

本实验通过 4 个子图展示 K-means 在不同数据分布下的表现：
  - kmeans01: 标准球形簇（K-means 最擅长的场景）
  - kmeans02: 各向异性（拉伸变形）的数据
  - kmeans03: 各类别样本数不均衡的数据
  - kmeans04: 从外部文件读取的真实聚类测试数据
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 设置中文字体，防止图表标题中的中文显示为方框
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """
    主函数：生成/加载四种不同场景的数据，进行 K-means 聚类并可视化
    """
    # 创建 2×2 的大画布
    plt.figure(figsize=(12, 12))

    n_samples = 150
    random_state = 170

    # ================================
    # 场景一（kmeans01）：标准球形簇
    # ================================
    print("=" * 50)
    print("场景一：标准球形簇")
    print("=" * 50)

    # make_blobs: sklearn 内置的聚类测试数据生成器
    #   n_samples=150   : 生成 150 个样本点
    #   random_state=170: 固定随机种子，保证每次运行结果一致
    x, y = make_blobs(n_samples=n_samples, random_state=random_state)
    print('x=', x, type(x), 'y=', y, type(y))

    # KMeans(n_clusters=3): 创建 K-means 模型，指定聚类数 K=3
    #   n_init=10 : 用 10 种不同随机初始化运行，选最好的结果
    #               （避免初始中心点不好导致陷入局部最优）
    #   fit_predict(x) : 聚类 + 返回每个点的簇标签
    y_pred = KMeans(n_clusters=3, n_init=10).fit_predict(x)
    print("y_pred : ", y_pred)

    # 子图 1 (2行2列中的第1个)
    plt.subplot(221)
    # scatter: 散点图，c=y_pred 表示颜色按簇标签着色
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.title("kmeans01 — 标准球形簇")

    # ================================
    # 场景二（kmeans02）：各向异性分布
    # ================================
    print("\n" + "=" * 50)
    print("场景二：各向异性分布")
    print("=" * 50)

    # 对数据进行线性变换（拉伸 + 旋转）
    # 将标准球形数据乘以一个变换矩阵，使簇变得细长且倾斜
    transformation = [[0.60834549, -0.63667341],
                      [-0.40887718, 0.85253229]]
    X_aniso = np.dot(x, transformation)  # 矩阵乘法实现线性变换

    # 用同样的 K-means 对变形后的数据聚类
    y_pred = KMeans(n_clusters=3, n_init=10).fit_predict(X_aniso)

    # 子图 2
    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("kmeans02 — 各向异性分布")

    # ================================
    # 场景三（kmeans03）：不均匀簇大小
    # ================================
    print("\n" + "=" * 50)
    print("场景三：不均匀簇大小")
    print("=" * 50)

    # 人为构造样本数不均衡的数据集：
    #   类别 0 取前 500，类别 1 只取 100，类别 2 取 200
    #   vstack: 垂直堆叠（沿行方向拼接）
    X_filtered = np.vstack((x[y == 0][:500],    # 类别0: 500个点
                            x[y == 1][:100],     # 类别1: 100个点
                            x[y == 2][:200]))    # 类别2: 200个点
    y_pred = KMeans(n_clusters=3, random_state=random_state, n_init=10).fit_predict(X_filtered)

    # 子图 3
    plt.subplot(223)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("kmeans03 — 不均匀簇大小")

    # ================================
    # 场景四（kmeans04）：外部真实数据
    # ================================
    print("\n" + "=" * 50)
    print("场景四：外部文件数据")
    print("=" * 50)

    try:
        dataMat = []
        # 从 data/testSet.txt 读取外部聚类测试数据
        # rb 模式: 以二进制读取，兼容不同换行符
        fr = open(DATA_DIR / "testSet.txt", "rb")
        for line in fr.readlines():
            # 跳过空行
            if line.decode("utf-8").strip() != "":
                # 解码为 utf-8 字符串，去除空白，按制表符切分
                curLine = line.decode("utf-8").strip().split('\t')
                # 将字符串转为浮点数
                fltLine = list(map(float, curLine))
                dataMat.append(fltLine)

        # 转换为 numpy 数组以便处理
        dataArray = np.array(dataMat)
        print("dataArray type = %s" % type(dataArray))

        # 用 K=5 对真实数据聚类（数据有 4-5 个自然簇）
        y_pred = KMeans(n_clusters=5, n_init=10).fit_predict(dataArray)

        # 子图 4
        plt.subplot(224)
        plt.scatter(dataArray[:, 0], dataArray[:, 1], c=y_pred)
        plt.title("kmeans04 — 外部文件 testSet.txt (K=5)")

    except Exception as e:
        print("\n未能加载或处理 testSet.txt (通常因为缺失该数据集)，原因:", e)

    # ================================
    # 保存并显示图表
    # ================================
    # 保存为 PNG 文件到 output 目录
    output_file = OUTPUT_DIR / "kmeans.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    print("聚类图已保存:", output_file.relative_to(PROJECT_ROOT))


# ================================
# 主程序入口
# ================================
if __name__ == '__main__':
    main()
