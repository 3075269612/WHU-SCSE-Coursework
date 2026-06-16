# 7.3.2 全连接神经网络案例 — MNIST 手写数字识别
# 本案例模拟银行支票、汇票等业务场景，使用全连接神经网络（多层感知机）
# 对 MNIST 数据集进行手写数字（0~9）的自动识别与分类。
#
# 完整流程包括以下四个核心环节：
#   7.3.1 数据处理   — 加载 MNIST 数据集、转换为张量、归一化、DataLoader 封装
#   7.3.2 模型搭建   — 定义三层全连接网络 (784→500→10)、前向传播、损失函数、反向传播
#   7.3.3 模型训练   — 批量梯度下降迭代优化，记录并打印训练日志
#   7.3.4 模型评估   — 在测试集上计算分类准确率
#
# 关键知识点回顾：
#   - 输入层: 784 个神经元（对应 28×28 像素的单通道灰度图像展开）
#   - 隐藏层: 500 个神经元 + ReLU 激活函数（引入非线性变换）
#   - 输出层: 10 个神经元（对应 0~9 十个数字类别）
#   - 损失函数: 交叉熵损失（CrossEntropyLoss），适用于多分类任务
#   - 优化器: 随机梯度下降（SGD）
#   - 训练轮次: 20 个 Epoch

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn           # 神经网络模块，提供各类层、损失函数等
import torch.optim as optim     # 优化器模块，提供 SGD、Adam 等
from torchvision import datasets, transforms  # 数据集加载与预处理
import torchvision
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# 1. 数据处理 (对应 7.3.1 节)
# ============================================================================

def load_mnist_data(batch_size=64):
    """
    加载 MNIST 数据集，并完成预处理：ToTensor + Normalize。
    MNIST 数据集包含：
      - 训练集: 60,000 张 28×28 的灰度手写数字图片
      - 测试集: 10,000 张 28×28 的灰度手写数字图片
      - 共 10 个类别 (0~9)

    :param batch_size: 每个批次的样本数量，默认 64
    :return: (train_loader, test_loader) 训练集和测试集的 DataLoader
    """
    # transforms.Compose 将多个预处理操作组合为一个流水线
    # ToTensor(): 将 PIL.Image 或 numpy.ndarray (H×W×C, 0~255)
    #             转换为 Tensor (C×H×W, 0~1 浮点数)
    # Normalize((0.1307,), (0.3081,)): 对单通道图像进行标准化
    #   mean=0.1307, std=0.3081 是 MNIST 数据集官方统计的均值和标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # datasets.MNIST 会自动检测 root 目录下是否已有数据，无则自动下载
    # train=True   → 加载训练集 (60,000 张)
    # train=False  → 加载测试集 (10,000 张)
    # download=not DATA_DIR.joinpath("MNIST").exists() → 已有本地数据时离线读取
    # shuffle=True  → 每个 epoch 随机打乱数据顺序，防止模型记忆数据排列规律
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=str(DATA_DIR),
                       train=True,
                       download=not (DATA_DIR / "MNIST").exists(),
                       transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=str(DATA_DIR),
                       train=False,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader


def imshow(img):
    """
    将归一化后的 Tensor 图像还原并显示。
    逆归一化公式: img = img * std + mean（此处还原到约 0~1 范围）
    :param img: 待显示的图像 Tensor
    """
    img = img / 2 + 0.5      # 逆归一化（简化为大致还原）
    npimg = img.numpy()       # Tensor → NumPy 数组
    # 由于 PyTorch 图像格式为 (C, H, W)，而 matplotlib 需要 (H, W, C)，故转置
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig(OUTPUT_DIR / "mnist_sample.png", dpi=150, bbox_inches="tight")
    plt.close()


def visualize_mnist_sample(train_loader):
    """
    随机抽取一个 batch 的数据并将其中的部分样本可视化，
    帮助直观了解数据集内容。
    """
    # 通过 iter() 获取 DataLoader 的迭代器，next() 取出一个 batch
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # torchvision.utils.make_grid 将多张图片拼成一张网格图
    img_grid = torchvision.utils.make_grid(images[:64])
    imshow(img_grid)
    print(f"对应标签: {labels[:10].tolist()}")


# ============================================================================
# 2. 模型搭建 (对应 7.3.2 节)
# ============================================================================

class NeuralNet(nn.Module):
    """
    三层全连接神经网络模型。
    结构: 输入层(784) → 隐藏层(500) + ReLU → 输出层(10)

    继承自 nn.Module，需要实现 __init__（定义层结构）和 forward（定义前向传播）。
    只要在 forward 中定义了计算流程，PyTorch 的 autograd 机制会自动推导 backward。
    """

    def __init__(self, input_num, hidden_num, output_num):
        """
        初始化网络各层。
        :param input_num:  输入层神经元数量 (MNIST: 28×28 = 784)
        :param hidden_num: 隐藏层神经元数量 (默认 500)
        :param output_num: 输出层神经元数量 (10 分类)
        """
        super(NeuralNet, self).__init__()
        # nn.Linear: 全连接层，执行 y = xW^T + b 的线性变换
        self.fc1 = nn.Linear(input_num, hidden_num)   # 第1层: 784 → 500
        self.fc2 = nn.Linear(hidden_num, output_num)  # 第2层: 500 → 10
        # ReLU 激活函数: 对隐藏层输出做非线性变换，打破线性组合的限制
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        定义前向传播的计算流程。
        :param x: 输入张量，形状为 (batch_size, 784)
        :return: 网络输出 logits，形状为 (batch_size, 10)
        """
        x = self.fc1(x)      # 线性变换: 784 → 500
        x = self.relu(x)     # 非线性激活
        y = self.fc2(x)      # 线性变换: 500 → 10（输出为各类别得分，即 logits）
        return y


def demo_forward_pass(model):
    """
    演示前向传播: 取少量样本输入网络，观察输出 logits 的形态。
    在训练前运行此函数，可验证网络结构是否正确。
    """
    print("\n>>> 前向传播示例 <<<")
    # 随机生成 2 个 "784 维" 的伪样本模拟输入
    demo_images = torch.randn(2, 784)
    demo_labels = torch.tensor([0, 6])  # 假设真实标签为 0 和 6

    print(f"输入形状: {demo_images.shape}")         # torch.Size([2, 784])
    print(f"真实标签: {demo_labels.tolist()}")

    out = model(demo_images)
    print(f"输出 logits 形状: {out.shape}")         # torch.Size([2, 10])
    print(f"输出 logits:\n{out}")
    # 注意: 此时的输出是未经过 Softmax 的原始得分（logits），
    # 数值大小不代表概率，仅表示各类别的相对得分。


def demo_loss_calculation(model):
    """
    演示交叉熵损失函数的计算过程。
    nn.CrossEntropyLoss 内部集成了 LogSoftmax + NLLLoss，
    因此模型输出无需手动添加 Softmax 层。
    """
    print("\n>>> 损失计算示例 <<<")
    criterion = nn.CrossEntropyLoss()

    demo_images = torch.randn(2, 784)
    demo_labels = torch.tensor([8, 7])

    out = model(demo_images)
    loss = criterion(out, demo_labels)
    print(f"输出 logits:\n{out}")
    print(f"交叉熵损失值: {loss.item():.4f}")
    # 损失值越大说明预测与真实标签的差距越大，训练目标就是最小化此值。


def demo_backward_pass(model):
    """
    演示反向传播与参数更新的完整流程。
    核心步骤：
      1. optimizer.zero_grad() — 清除上一次迭代的梯度（PyTorch 默认累积梯度）
      2. loss.backward()       — 基于损失值自动计算所有参数的梯度
      3. optimizer.step()      — 根据梯度更新参数
    """
    print("\n>>> 反向传播与参数更新示例 <<<")
    # 使用 SGD 优化器，学习率 lr 控制每次参数更新的步幅
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    demo_images = torch.randn(2, 784)
    demo_labels = torch.tensor([8, 7])

    # --- 一次完整的参数更新循环 ---
    optimizer.zero_grad()          # 步骤1: 清除累积梯度
    out = model(demo_images)       # 步骤2: 前向传播
    loss = criterion(out, demo_labels)  # 步骤3: 计算损失
    loss.backward()                # 步骤4: 反向传播——自动计算梯度
    optimizer.step()               # 步骤5: 更新参数 weight = weight - lr * gradient

    print(f"更新后的损失值: {loss.item():.4f}")


# ============================================================================
# 3. 模型训练 (对应 7.3.3 节)
# ============================================================================

def train(model, train_loader, criterion, optimizer, epoches):
    """
    在训练集上对模型进行多轮迭代训练。
    每一轮（Epoch）遍历全部训练数据一次，每次取一个 batch 做前向/反向传播。

    :param model:       待训练的神经网络模型
    :param train_loader: 训练数据的 DataLoader
    :param criterion:   损失函数
    :param optimizer:   优化器
    :param epoches:     训练轮数（Epoch 数量）
    """
    model.train()  # 设置为训练模式（影响 Dropout、BatchNorm 等层的行
    for epoch in range(epoches):
        for i, data in enumerate(train_loader):
            (images, labels) = data
            # MNIST 原始图像形状为 (batch, 1, 28, 28)，
            # 全连接网络需要先将其展平为 (batch, 784) 的一维向量
            images = images.reshape(-1, 28 * 28)

            # ---- 标准训练四步曲 ----
            output = model(images)               # (1) 前向传播
            loss = criterion(output, labels)     # (2) 计算损失
            optimizer.zero_grad()                # (3) 清除梯度
            loss.backward()                      # (4) 反向传播
            optimizer.step()                     # (5) 参数更新

            # 每 100 个 batch 打印一次日志，监控训练进展
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epoches} - {i + 1}], Loss: {loss.item():.4f}')


# ============================================================================
# 4. 模型评估 (对应 7.3.4 节)
# ============================================================================

def evaluate(model, test_loader):
    """
    在测试集上评估模型的分类准确率。
    使用 torch.no_grad() 上下文管理器禁用梯度计算，节省显存并加速推理。

    :param model:      已训练的模型
    :param test_loader: 测试数据的 DataLoader
    :return: 分类准确率 (0~1 之间的小数)
    """
    model.eval()  # 设置为评估模式（关闭 Dropout 等仅在训练时生效的层）
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用 autograd，不构建计算图
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28)
            output = model(images)

            # torch.max(output, 1) 返回 (最大值, 最大值索引)
            # dim=1 表示沿第 2 维（类别维度）取最大值
            _, predicted = torch.max(output, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'\n测试集共 {total} 张图片，分类准确率: {accuracy:.2f}%')
    return accuracy


# ============================================================================
# 5. 主函数: 串联完整流程
# ============================================================================

def main():
    """
    主函数：按顺序执行数据处理 → 模型搭建 → 模型训练 → 模型评估 的完整流程。
    """
    print("=" * 60)
    print("7.3.2 MNIST 手写数字识别 — 全连接神经网络")
    print("=" * 60)

    # ---- 超参数配置 ----
    epoches = 20       # 训练轮数（遍历整个训练集的次数）
    lr = 0.001         # 学习率（控制参数更新步幅）
    input_num = 784    # 输入维度: 28×28 像素展开
    hidden_num = 500   # 隐藏层神经元数量
    output_num = 10    # 输出类别数: 0~9
    batch_size = 64    # 每批样本数
    # 检测是否有可用的 GPU，有则使用 CUDA 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ---- (1) 数据处理 ----
    print("\n>>> 加载 MNIST 数据集...")
    train_loader, test_loader = load_mnist_data(batch_size)
    print("数据集加载完成！")
    # 可视化一批样本（可选，注释掉可加快运行）
    # visualize_mnist_sample(train_loader)

    # ---- (2) 模型搭建 ----
    print("\n>>> 搭建神经网络模型...")
    model = NeuralNet(input_num, hidden_num, output_num)
    model = model.to(device)  # 将模型迁移到对应设备
    print(model)              # 打印模型结构概览

    # 定义损失函数和优化器
    # CrossEntropyLoss: 交叉熵损失，内部自动做 Softmax → Log → NLLLoss
    criterion = nn.CrossEntropyLoss()
    # SGD: 随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # ---- 用于演示的小实验（可选） ----
    # demo_forward_pass(model)
    # demo_loss_calculation(model)
    # demo_backward_pass(model)

    # ---- (3) 模型训练 ----
    print(f"\n>>> 开始训练 (共 {epoches} 轮)...")
    train(model, train_loader, criterion, optimizer, epoches)
    print("训练完成！")

    # ---- (4) 模型评估 ----
    print("\n>>> 在测试集上评估模型...")
    evaluate(model, test_loader)


if __name__ == '__main__':
    main()
