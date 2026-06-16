# 7.3.1 激活函数可视化案例
# 本案例演示神经网络中四种常用激活函数的定义、计算与图形绘制：
#   Sigmoid 函数 — 将输出映射到 [0,1]，适用于二分类输出层
#   ReLU 函数   — 缓解梯度消失，广泛用于隐藏层
#   Tanh 函数   — 输出映射到 [-1,1]，均值为0，收敛速度较快
#   Softmax 函数— 将得分转化为概率分布，适用于多分类输出层
# 注意：7.3.1 对应该章节「设计与实现」中第一小节内容，
# 本章核心案例（MNIST手写数字识别）的完整代码见 7.3.2 文件。


import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import numpy as np
import matplotlib
matplotlib.use('Agg')

# ---- 配置中文字体支持 ----
# 强制重建字体缓存，确保中文能正确显示
import matplotlib.font_manager as fm
# 清除已有字体缓存，确保新设置的字体生效
fm._load_fontmanager(try_read_cache=False)
# 微软雅黑是 Windows 下最常用的中文字体，Noto Sans SC 为开源备选
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. Sigmoid 函数
# ============================================================================
# 公式: sigmoid(x) = 1 / (1 + exp(-x))
# 特点: 输出范围 (0, 1)，平滑单调，但两端饱和区梯度趋近于 0，容易导致梯度消失。

def sigmoid(x):
    """
    Sigmoid 激活函数。
    :param x: 输入值，可以是标量或 NumPy 数组
    :return: Sigmoid 函数输出
    """
    return 1 / (1 + np.exp(-x))


def plot_sigmoid():
    """
    绘制 Sigmoid 函数及其导数图像。
    """
    # 在 [-5, 5] 区间生成 1000 个均匀分布的数据点，使曲线更加平滑
    x = np.linspace(-5.0, 5.0, 1000)
    y = sigmoid(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='Sigmoid', linewidth=2, color='#2196F3')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    # 指定 y 轴范围以突出 [0, 1] 区间
    plt.ylim(-0.1, 1.1)
    plt.title('Sigmoid 激活函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sigmoid.png', dpi=150)
    plt.close()


def demo_sigmoid():
    """
    演示 Sigmoid 函数对 NumPy 数组的向量化处理能力。
    """
    print(">>> Sigmoid 函数示例 <<<")
    # 使用数组演示——Sigmoid 函数自动对每个元素独立计算
    x = np.array([-1.0, 1.0, 2.0])
    res = sigmoid(x)
    print(f"输入: {x}")
    print(f"输出: {res}")
    print()


# ============================================================================
# 2. ReLU 函数
# ============================================================================
# 公式: ReLU(x) = max(0, x)
# 特点: 正半轴梯度恒为 1，缓解梯度消失；负半轴输出为 0，带来稀疏性。

def relu(x):
    """
    ReLU (Rectified Linear Unit) 激活函数。
    使用 NumPy 的 maximum 函数，从 0 和 x 中逐元素取较大值。
    :param x: 输入值，可以是标量或 NumPy 数组
    :return: ReLU 函数输出
    """
    return np.maximum(0, x)


def plot_relu():
    """
    绘制 ReLU 函数图像。
    """
    x = np.linspace(-6.0, 6.0, 1000)
    y = relu(x)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label='ReLU', linewidth=2, color='#4CAF50')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    plt.title('ReLU 激活函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('relu(x)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'relu.png', dpi=150)
    plt.close()


# ============================================================================
# 3. Tanh 函数
# ============================================================================
# 使用 PyTorch 内置的 tanh 函数进行演示。

def plot_tanh():
    """
    绘制 Tanh（双曲正切）函数图像。
    使用 PyTorch 框架中的 torch.nn.functional.tanh 函数。
    """
    # 在 [-6, 6] 区间生成数据点
    x = np.linspace(-6.0, 6.0, 1000)
    # 将 NumPy 数组转换为 PyTorch Tensor
    input_tensor = torch.tensor(x, dtype=torch.float32)
    # 调用 PyTorch 内置的 Tanh 激活函数
    output_tensor = torch.nn.functional.tanh(input_tensor)

    plt.figure(figsize=(8, 4))
    plt.plot(x, output_tensor.numpy(), label='Tanh', linewidth=2, color='#FF9800')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    plt.ylim(-1.1, 1.1)
    plt.title('Tanh 激活函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('tanh(x)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'tanh.png', dpi=150)
    plt.close()


# ============================================================================
# 4. Softmax 函数
# ============================================================================
# Softmax 将任意实数值向量映射为概率分布（和为 1），常用于多分类输出层。

def plot_softmax():
    """
    绘制 Softmax 函数对不同输入值的归一化输出。
    使用 PyTorch 框架中的 torch.nn.functional.softmax 函数。
    注意：Softmax 本质上是向量函数，此处沿 dim=0 对所有数据点做全局 Softmax，
    以直观展示其概率归一化特性。
    """
    # 在 [-10, 10] 区间生成数据点
    x = np.linspace(-10.0, 10.0, 1000)
    input_tensor = torch.tensor(x, dtype=torch.float32)
    # Softmax 沿 dim=0 计算，输出为 [0,1] 之间的概率值，总和为 1
    output_tensor = torch.nn.functional.softmax(input_tensor, dim=0)

    plt.figure(figsize=(8, 4))
    plt.plot(x, output_tensor.numpy(), label='Softmax (dim=0)', linewidth=2, color='#E91E63')
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.title('Softmax 激活函数', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('softmax(x)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'softmax.png', dpi=150)
    plt.close()


# ============================================================================
# 5. 四种激活函数综合对比
# ============================================================================

def plot_all_activations():
    """
    将四种激活函数绘制在同一张图上，便于直观对比各自的形状与特性。
    """
    x = np.linspace(-6.0, 6.0, 1000)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    y_sigmoid = sigmoid(x)
    y_relu = relu(x)
    y_tanh = torch.nn.functional.tanh(x_tensor).numpy()
    y_softmax = torch.nn.functional.softmax(x_tensor, dim=0).numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_sigmoid, label='Sigmoid', linewidth=2)
    plt.plot(x, y_relu, label='ReLU', linewidth=2)
    plt.plot(x, y_tanh, label='Tanh', linewidth=2)
    plt.plot(x, y_softmax, label='Softmax (dim=0)', linewidth=2)

    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.ylim(-1.1, 1.3)
    plt.title('四种激活函数对比', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('output')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'all_activations.png', dpi=150)
    plt.close()


def main():
    """
    主函数：依次展示四种激活函数的数值计算示例与图像。
    """
    print("=" * 60)
    print("7.3.1 神经网络激活函数可视化")
    print("=" * 60)

    # ---- Sigmoid ----
    demo_sigmoid()
    print("正在绘制 Sigmoid 函数图像...")
    plot_sigmoid()

    # ---- ReLU ----
    print("正在绘制 ReLU 函数图像...")
    plot_relu()

    # ---- Tanh ----
    print("正在绘制 Tanh 函数图像...")
    plot_tanh()

    # ---- Softmax ----
    print("正在绘制 Softmax 函数图像...")
    plot_softmax()

    # ---- 综合对比 ----
    print("正在绘制四种激活函数综合对比图...")
    plot_all_activations()

    print("\n所有激活函数可视化完成！图像已保存至 output/ 目录。")


if __name__ == '__main__':
    main()
