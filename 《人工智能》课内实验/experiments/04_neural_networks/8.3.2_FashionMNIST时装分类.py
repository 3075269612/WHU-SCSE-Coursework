# 8.3.2 卷积神经网络案例 — FashionMNIST 时装分类
# 本案例使用 PyTorch 框架搭建卷积神经网络（CNN），对 FashionMNIST 数据集
# 进行 10 类时尚物品的自动分类。
#
# FashionMNIST 数据集简介:
#   - 共 10 个类别: T恤/上衣、裤子、套头衫、连衣裙、外套、凉鞋、
#                   衬衫、运动鞋、包、踝靴
#   - 每张图片: 28×28 像素灰度图
#   - 训练集: 60,000 张 | 测试集: 10,000 张
#   - 相比 MNIST 手写数字, 时尚物品之间的视觉差异更细微, 分类难度更高
#
# CNN 相比全连接网络的三大核心优势:
#   1. 局部感知   — 卷积核每次只关注局部区域（感受野），模拟生物视觉
#   2. 权值共享   — 同一卷积核在整个图像上共享参数，大幅减少参数量
#   3. 空间降采样 — 池化层逐步缩小特征图，保留关键特征的同时降低计算量
#
# 完整流程包括:
#   8.3.1 准备数据 — 加载 FashionMNIST、数据变换、DataLoader 封装
#   8.3.2 搭建模型 — 2 层卷积 + 2 层全连接的 CNN
#   8.3.3 训练模型 — Adam 优化器 + 交叉熵损失, GPU 加速训练
#   8.3.4 测试模型 — 测试集准确率评估

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn           # 神经网络模块
import torch.optim as optim     # 优化器模块
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
# 1. 数据准备 (对应 8.3.1 节)
# ============================================================================

def setup_device():
    """
    配置计算设备: GPU > CPU。
    PyTorch 中使用 device 对象统一管理模型和数据所在的设备。
    """
    # 方式1: 使用 torch.device（推荐，便于后续 .to(device) 调用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 方式2: 通过环境变量指定 GPU（二选一即可）
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    return device


def get_dataloaders(batch_size=256, num_workers=0):
    """
    加载 FashionMNIST 数据集并封装为 DataLoader。

    FashionMNIST 10 个类别标签对应关系:
      0: T恤/上衣 (T-shirt/top)
      1: 裤子 (Trouser)
      2: 套头衫 (Pullover)
      3: 连衣裙 (Dress)
      4: 外套 (Coat)
      5: 凉鞋 (Sandal)
      6: 衬衫 (Shirt)
      7: 运动鞋 (Sneaker)
      8: 包 (Bag)
      9: 踝靴 (Ankle boot)

    :param batch_size: 每批样本数量，默认 256
    :param num_workers: 数据加载子进程数，Windows 下需设为 0
    :return: (train_loader, test_loader) 训练集与测试集 DataLoader
    """
    # 数据预处理流水线:
    #   ToTensor():   将 PIL.Image (0~255) 转为 Tensor (0~1)
    #   Normalize:    标准化，均值和标准差设为 0.5，将像素值映射到 [-1, 1]
    #                 公式: (x - 0.5) / 0.5 = 2x - 1
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载训练集 (train=True)
    # root=DATA_DIR: 数据集存放目录；已有本地数据时离线读取
    train_data = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=True,
        download=not (DATA_DIR / "FashionMNIST").exists(),
        transform=data_transform
    )

    # 加载测试集 (train=False)
    test_data = datasets.FashionMNIST(
        root=str(DATA_DIR),
        train=False,
        download=not (DATA_DIR / "FashionMNIST").exists(),
        transform=data_transform
    )

    # DataLoader 将数据集封装为可迭代的批次加载器
    # shuffle=True:  训练时打乱数据顺序，防止模型记忆排列规律
    # shuffle=False: 测试时无需打乱，保持顺序便于分析
    # drop_last=True: 丢弃最后一个不足 batch_size 的 batch（避免因 batch 大小不一致导致的 batch norm 问题）
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def show_sample(train_loader):
    """
    从训练集中取一个 batch 并展示第一张图片及其标签，
    帮助直观了解 FashionMNIST 数据内容。
    """
    # FashionMNIST 类别名称映射表
    class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
                   '凉鞋', '衬衫', '运动鞋', '包', '踝靴']

    # 通过 iter + next 获取一个 batch
    image, label = next(iter(train_loader))
    print(f"Batch 图像形状: {image.shape}")  # (256, 1, 28, 28)
    print(f"Batch 标签形状: {label.shape}")  # (256,)

    # 显示第一张图片（灰度图，cmap='gray'）
    plt.imshow(image[0][0].numpy(), cmap="gray")
    plt.title(f"标签: {label[0].item()} ({class_names[label[0].item()]})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fashionmnist_sample.png', dpi=150)
    plt.close()


# ============================================================================
# 2. 模型搭建 (对应 8.3.2 节)
# ============================================================================

class Net(nn.Module):
    """
    用于 FashionMNIST 分类的卷积神经网络。

    网络架构:
      ┌─────────────────────────────────────────────┐
      │ 卷积块 (self.conv):                          │
      │   Conv2d(1, 32, 5)  → ReLU → MaxPool2d(2,2) │
      │   → Dropout(0.3)                             │
      │   Conv2d(32, 64, 5) → ReLU → MaxPool2d(2,2) │
      │   → Dropout(0.3)                             │
      ├─────────────────────────────────────────────┤
      │ 全连接块 (self.fc):                           │
      │   Linear(64*4*4, 512) → ReLU                 │
      │   → Linear(512, 10)                           │
      └─────────────────────────────────────────────┘

    维度变化追踪 (输入 batch, 1, 28, 28):
      Conv1(1→32, 5×5)  → (batch, 32, 24, 24)   [28-5+1=24]
      MaxPool(2×2, s=2) → (batch, 32, 12, 12)   [24/2=12]
      Conv2(32→64, 5×5) → (batch, 64, 8, 8)     [12-5+1=8]
      MaxPool(2×2, s=2) → (batch, 64, 4, 4)     [8/2=4]
      Flatten            → (batch, 64*4*4=1024)
      Linear(1024→512)   → (batch, 512)
      Linear(512→10)     → (batch, 10)           [10 个类别的 logits]
    """

    def __init__(self):
        super(Net, self).__init__()

        # ---- 卷积块 ----
        # nn.Sequential 将多个层按顺序组合为一个模块，简化 forward 编写
        self.conv = nn.Sequential(
            # 第 1 卷积层: 输入 1 通道 (灰度图), 输出 32 通道, 5×5 卷积核
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),                    # 非线性激活
            nn.MaxPool2d(2, stride=2),    # 2×2 最大池化, 步幅 2 (空间尺寸减半)
            nn.Dropout(0.3),              # 随机丢弃 30% 神经元, 防止过拟合
            # 第 2 卷积层: 输入 32 通道, 输出 64 通道, 5×5 卷积核
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),    # 空间尺寸再减半
            nn.Dropout(0.3)
        )

        # ---- 全连接块 ----
        self.fc = nn.Sequential(
            # 全连接层 1: 将展平后的特征向量映射到 512 维隐藏层
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            # 全连接层 2 (输出层): 输出 10 个类别的原始得分 (logits)
            # 注意: 这里不加 Softmax，因为 CrossEntropyLoss 内部自带 Softmax
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        定义前向传播。
        :param x: 输入图像 Tensor，形状 (batch, 1, 28, 28)
        :return:  分类 logits，形状 (batch, 10)
        """
        x = self.conv(x)                    # 卷积特征提取
        x = x.view(-1, 64 * 4 * 4)         # 展平: (batch, 64, 4, 4) → (batch, 1024)
        x = self.fc(x)                      # 全连接分类
        return x


def print_model_info(model):
    """
    打印模型结构信息。
    """
    print(model)
    # 统计模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


# ============================================================================
# 3. 模型训练 (对应 8.3.3 节)
# ============================================================================

def train(model, train_loader, criterion, optimizer, epoch, device):
    """
    执行一个 Epoch 的训练。
    训练模式 (model.train()) 会启用 Dropout、BatchNorm 等仅在训练时有效的层。

    :param model:       神经网络模型
    :param train_loader: 训练数据 DataLoader
    :param criterion:   损失函数
    :param optimizer:   优化器
    :param epoch:       当前轮次编号
    :param device:      计算设备 (CPU/GPU)
    """
    model.train()  # 切换到训练模式
    train_loss = 0  # 累积损失，用于计算平均损失

    for data, label in train_loader:
        # 将数据迁移到指定设备（GPU 或 CPU）
        data, label = data.to(device), label.to(device)

        # ---- 标准训练五步曲 ----
        optimizer.zero_grad()           # (1) 清除上一次迭代的梯度
        output = model(data)            # (2) 前向传播，计算预测值
        loss = criterion(output, label) # (3) 计算损失（预测值与真实值的差距）
        loss.backward()                 # (4) 反向传播，自动计算所有参数梯度
        optimizer.step()                # (5) 根据梯度更新参数

        # 累积损失 (乘以 batch 大小是为了最后求平均时加权正确)
        train_loss += loss.item() * data.size(0)

    # 计算整个训练集的平均损失
    train_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f}')


# ============================================================================
# 4. 模型测试/评估 (对应 8.3.4 节)
# ============================================================================

def validate(model, test_loader, criterion, epoch, device):
    """
    在测试集上评估模型性能。
    评估模式 (model.eval()) 会关闭 Dropout、固定 BatchNorm 参数等。

    与训练函数的主要区别:
      1. 模型状态: model.eval() vs model.train()
      2. 不需要 optimizer（不更新参数）
      3. 不需要 loss.backward()（不计算梯度）
      4. 使用 torch.no_grad() 禁用 autograd，节省显存并加速

    :param model:       神经网络模型
    :param test_loader:  测试数据 DataLoader
    :param criterion:   损失函数
    :param epoch:       当前轮次编号
    :param device:      计算设备
    """
    model.eval()  # 切换到评估模式
    val_loss = 0
    gt_labels = []    # 真实标签列表
    pred_labels = []  # 预测标签列表

    with torch.no_grad():  # 禁用梯度计算
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)

            output = model(data)                          # 前向传播
            preds = torch.argmax(output, dim=1)           # 取最大 logit 对应的类别索引

            # 收集真实标签和预测标签（用于计算准确率）
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())

            # 计算损失
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)

    val_loss = val_loss / len(test_loader.dataset)

    # 将所有 batch 的标签拼接为完整数组
    gt_labels = np.concatenate(gt_labels)
    pred_labels = np.concatenate(pred_labels)

    # 计算分类准确率
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)

    print(f'Epoch: {epoch} \tValidation Loss: {val_loss:.6f}, Accuracy: {acc:.6f}')
    return acc


# ============================================================================
# 5. 模型保存与加载
# ============================================================================

def save_model(model, save_path="./FashionModel.pkl"):
    """
    保存训练好的模型。
    torch.save 可以保存整个模型（含结构和参数），也可以只保存参数字典。
    """
    torch.save(model, save_path)
    print(f"模型已保存至: {save_path}")


def load_model(save_path, device):
    """
    加载已保存的模型。
    map_location 参数确保模型能被加载到正确的设备上。
    """
    model = torch.load(save_path, map_location=torch.device(device))
    print(f"模型已从 {save_path} 加载")
    return model


# ============================================================================
# 6. 主函数: 串联完整流程
# ============================================================================

def main():
    """
    主函数: 执行 FashionMNIST 时装分类的完整 CNN 流程。
    """
    print("=" * 60)
    print("8.3.2 FashionMNIST 时装分类 — 卷积神经网络")
    print("=" * 60)

    # ---- 超参数配置 ----
    batch_size = 256     # 批大小: 每次用 256 张图片更新参数
    num_workers = 0      # Windows 下必须设为 0, 否则多线程加载会报错
    lr = 1e-4            # 学习率 (Adam 优化器通常用较小学习率)
    epochs = 20          # 训练轮数
    save_path = "./FashionModel.pkl"  # 模型保存路径

    # ---- (1) 设备配置 ----
    device = setup_device()

    # ---- (2) 数据准备 ----
    print("\n>>> 加载 FashionMNIST 数据集...")
    train_loader, test_loader = get_dataloaders(batch_size, num_workers)
    print("数据集加载完成！")
    print(f"训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")

    # 可视化数据样本（可选）
    # show_sample(train_loader)

    # ---- (3) 模型搭建 ----
    print("\n>>> 搭建 CNN 模型...")
    model = Net()
    model = model.to(device)  # 将模型迁移到 GPU / CPU
    print_model_info(model)

    # ---- (4) 损失函数与优化器 ----
    # 交叉熵损失: 适用于多分类任务，PyTorch 内部自动进行 Softmax → Log → NLLLoss
    criterion = nn.CrossEntropyLoss()
    # Adam 优化器: 融合了动量梯度下降 + RMSProp 的自适应学习率算法
    #   优点: 超参数相对固定、收敛快、适用于大规模数据和稀疏梯度
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ---- (5) 训练与验证循环 ----
    print(f"\n>>> 开始训练 (共 {epochs} 轮, 学习率={lr})...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, device)
        acc = validate(model, test_loader, criterion, epoch, device)

        # 记录最佳准确率
        if acc > best_acc:
            best_acc = acc

    print(f"\n训练完成！最佳测试准确率: {best_acc:.4f}")

    # ---- (6) 保存模型 ----
    save_model(model, save_path)

    print("\n" + "=" * 60)
    print("FashionMNIST 时装分类完整流程执行完毕！")
    print("=" * 60)


if __name__ == '__main__':
    main()
