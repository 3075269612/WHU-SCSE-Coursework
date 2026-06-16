# 8.3.1 CNN 基础组件底层实现案例
# 本案例不使用 PyTorch 等高级框架的现成卷积层，而是基于 NumPy 从零实现
# 卷积神经网络的核心组件，以深入理解 CNN 的内部工作机制。
#
# 内容包括:
#   1. im2col 函数      — 将四维输入数据展开为二维矩阵，简化卷积运算
#   2. col2im 函数      — im2col 的逆操作，用于反向传播
#   3. Convolution 类   — 卷积层的前向传播与反向传播
#   4. Pooling 类       — 池化层（最大值池化）的前向传播与反向传播
#   5. Relu 类          — ReLU 激活函数层
#   6. Affine 类        — 全连接层（仿射变换）
#   7. SoftmaxWithLoss 类 — Softmax + 交叉熵损失的组合层
#   8. SimpleConvNet 类 — 将以上组件组装为完整的 CNN 网络
#
# 关键概念:
#   - 四维数据格式: (batch_num, channel, height, width) 简称 NCHW
#   - im2col 技巧: 将滑窗操作转化为矩阵乘法，充分利用 NumPy 的并行计算能力
#   - 权值共享: 卷积核在整个输入上滑动，参数量远小于全连接
#   - 感受野: 卷积核每次只关注局部区域，模拟生物视觉系统

from collections import OrderedDict
import numpy as np


# ============================================================================
# 1. im2col 函数 — 将图像数据展开为列矩阵
# ============================================================================
# 为什么需要 im2col？
#   卷积本质上是滤波器在输入上滑窗做乘积累加。如果直接用 for 循环实现，
#   效率极低（NumPy 中逐元素访问很慢）。im2col 将滑窗操作转化为一次矩阵乘法，
#   利用 BLAS 级别的优化大幅加速。
#
# 原理说明：
#   输入形状 (N, C, H, W) → 在每个空间位置取出一个 filter_h×filter_w 的窗 →
#   展开为 (N*out_h*out_w, C*filter_h*filter_w) 的二维矩阵。
#   滤波器展开为 (C*filter_h*filter_w, FN) 后，两者做矩阵乘法即可得到卷积结果。

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将四维输入数据 (N, C, H, W) 展开为二维矩阵，便于卷积层进行矩阵乘法。

    :param input_data: 四维数组，形状为 (N, C, H, W)
                       N  — 批量大小 (batch size)
                       C  — 通道数 (channels)
                       H  — 图像高度 (height)
                       W  — 图像宽度 (width)
    :param filter_h:   滤波器的高度
    :param filter_w:   滤波器的宽度
    :param stride:     卷积步幅，默认为 1
    :param pad:        填充大小，默认为 0
    :return: 展开后的二维矩阵，形状为 (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape

    # 计算输出特征图的空间尺寸
    # 公式: out_size = (input_size + 2*pad - filter_size) // stride + 1
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # np.pad 对输入数据的 H 和 W 维度进行零填充
    # pad 参数格式: [(前,后), (前,后), ...] 对每个轴的填充
    # 这里只对高和宽两个空间维度做填充，N 和 C 维度不填充
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    # 预分配一个六维数组，用于存放每个滑窗位置的数据
    # 形状: (N, C, filter_h, filter_w, out_h, out_w)
    # filter_h 和 filter_w 是每个窗内部的行列，out_h 和 out_w 是窗的滑行位置
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 遍历滤波器的每个位置，从 img 中取出对应的滑窗数据
    for y in range(filter_h):
        y_max = y + stride * out_h          # 该行对应的最大高度索引
        for x in range(filter_w):
            x_max = x + stride * out_w      # 该列对应的最大宽度索引
            # 步进切片: y:y_max:stride 表示从 y 开始到 y_max，每隔 stride 取一个
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # transpose 重排维度后再 reshape 为二维矩阵
    # (N, C, FH, FW, OH, OW) → (N, OH, OW, C, FH, FW) → (N*OH*OW, C*FH*FW)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


# ============================================================================
# 2. col2im 函数 — im2col 的逆操作
# ============================================================================
# 反向传播时需要将二维梯度矩阵还原为四维形状，
# 以便将梯度传递回上一层。col2im 就是 im2col 的逆向过程。

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将展开后的二维矩阵还原为四维数组，是 im2col 的逆操作。
    用于卷积层反向传播时将梯度还原为输入形状。

    :param col:          展开后的二维矩阵
    :param input_shape:  原始输入的形状 (N, C, H, W)
    :param filter_h:     滤波器高度
    :param filter_w:     滤波器宽度
    :param stride:       步幅
    :param pad:          填充
    :return: 还原后的四维数组，形状与 input_shape 一致
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 先 reshape 再 transpose 还原为六维数组
    # (N*OH*OW, C*FH*FW) → (N, OH, OW, C, FH, FW) → (N, C, FH, FW, OH, OW)
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # 创建比原始图像略大的零数组，以容纳反向传播时可能的越界索引
    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))

    # 将 col 中各窗的梯度累加回对应位置
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # 注意这里用 += 累加，因为同一个像素可能被多个窗覆盖
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 裁剪掉填充部分，恢复原始尺寸
    return img[:, :, pad:H + pad, pad:W + pad]


# ============================================================================
# 3. Convolution 类 — 卷积层
# ============================================================================

class Convolution:
    """
    卷积层的完整实现，包含前向传播和反向传播。

    前向传播流程:
      输入 x (N,C,H,W) → im2col 展开 → 与滤波器矩阵相乘 → 加偏置 → reshape 输出

    反向传播流程:
      计算 dW (滤波器梯度)、db (偏置梯度)、dx (传给上一层的梯度)

    参数说明:
      W: 滤波器权重，形状为 (FN, C, FH, FW)
         FN = 滤波器数量（输出通道数）、C = 输入通道数
         FH = 滤波器高度、FW = 滤波器宽度
      b: 偏置，形状为 (FN,)，每个输出通道一个偏置值
    """

    def __init__(self, W, b, stride=1, pad=0):
        """
        初始化卷积层。
        :param W:      权重矩阵，形状 (FN, C, FH, FW)
        :param b:      偏置向量，形状 (FN,)
        :param stride: 步幅
        :param pad:    填充
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # ---- 中间数据，反向传播时使用 ----
        self.x = None       # 前向传播时的输入
        self.col = None     # im2col 展开后的矩阵
        self.col_W = None   # 滤波器展开后的矩阵 (C*FH*FW, FN)

        # ---- 权重和偏置的梯度 ----
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        卷积层前向传播。
        :param x: 输入四维数组 (N, C, H, W)
        :return:  输出四维数组 (N, FN, OH, OW)
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # 计算输出特征图的尺寸
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        # im2col 展开: (N,C,H,W) → (N*OH*OW, C*FH*FW)
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 滤波器展开: (FN, C, FH, FW) → (C*FH*FW, FN)
        col_W = self.W.reshape(FN, -1).T

        # 矩阵乘法 + 偏置: (N*OH*OW, FN)
        out = np.dot(col, col_W) + self.b

        # reshape 回四维: (N*OH*OW, FN) → (N, OH, OW, FN) → (N, FN, OH, OW)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 保存中间变量供反向传播使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        """
        卷积层反向传播。
        :param dout: 来自后一层的梯度，形状 (N, FN, OH, OW)
        :return:     传递给前一层的梯度 dx，形状 (N, C, H, W)
        """
        FN, C, FH, FW = self.W.shape

        # 将 dout 转换为二维: (N, FN, OH, OW) → (N*OH*OW, FN)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 偏置梯度: 对 batch 维度求和
        self.db = np.sum(dout, axis=0)
        # 权重梯度: col^T × dout → (C*FH*FW, FN) → (FN, C, FH, FW)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 传给输入层的梯度: dout × col_W^T
        dcol = np.dot(dout, self.col_W.T)
        # 还原为四维
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


# ============================================================================
# 4. Pooling 类 — 池化层（最大值池化）
# ============================================================================

class Pooling:
    """
    最大值池化层 (Max Pooling) 的完整实现。

    池化层与卷积层的重要区别:
      1. 没有可学习的参数（无需权重/偏置）
      2. 通道数不变: 输入 C 个通道，输出仍是 C 个通道
      3. 池化是逐通道独立进行的，各通道之间互不干扰
      4. 对微小位置变化具有鲁棒性（平移不变性）

    实现流程:
      展开输入 → 对每个池化窗口取最大值 → 转换为合适的输出形状
    """

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        """
        :param pool_h: 池化窗口高度（通常为 2）
        :param pool_w: 池化窗口宽度（通常为 2）
        :param stride:  步幅（通常与窗口大小相同，即不重叠池化）
        :param pad:     填充（池化一般不用填充）
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None        # 前向传播输入
        self.arg_max = None  # 记录最大值的位置索引（反向传播时需要）

    def forward(self, x):
        """
        池化层前向传播。
        :param x: 输入四维数组 (N, C, H, W)
        :return:  输出四维数组 (N, C, OH, OW)
        """
        N, C, H, W = x.shape

        # 计算输出尺寸
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 步骤1: 展开输入数据
        # im2col 展开 → (N*OH*OW, C*pool_h*pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # 重塑使得每行对应一个池化窗口的所有元素
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 步骤2: 求每个窗口（每行）的最大值索引和最大值
        arg_max = np.argmax(col, axis=1)  # 记录索引（用于反向传播）
        out = np.max(col, axis=1)          # 取最大值

        # 步骤3: 转换为合适的四维输出形状
        # (N*OH*OW*C,) → (N, OH, OW, C) → (N, C, OH, OW)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        """
        池化层反向传播 — 只将梯度传给前向传播时最大值所在的位置，
        其余位置的梯度为 0（因为非最大值对输出没有贡献）。
        """
        # (N, C, OH, OW) → (N, OH, OW, C)
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # 创建一个全零数组，在 arg_max 记录的位置填入 dout 的梯度值
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        # 使用 col2im 将梯度还原为输入形状
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


# ============================================================================
# 5. Relu 类 — ReLU 激活函数层
# ============================================================================

class Relu:
    """
    ReLU 激活函数层。
    前向: out = max(0, x)
    反向: 输入 > 0 的位置梯度不变，≤ 0 的位置梯度为 0
    """

    def __init__(self):
        self.mask = None  # 记录哪些位置的输入 ≤ 0（True 表示需要屏蔽）

    def forward(self, x):
        """
        前向传播: 执行 ReLU 激活。
        mask 记录了 x ≤ 0 的位置，反向传播时这些位置的梯度置零。
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        """反向传播: 将 x ≤ 0 位置的梯度置零"""
        dout[self.mask] = 0
        dx = dout
        return dx


# ============================================================================
# 6. Affine 类 — 全连接层（仿射变换）
# ============================================================================

class Affine:
    """
    全连接层（仿射变换层），执行 y = x·W + b 的线性变换。
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """前向传播"""
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)  # 展平为二维
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        """反向传播"""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原为输入形状
        return dx


# ============================================================================
# 7. SoftmaxWithLoss 类 — Softmax + 交叉熵损失组合层
# ============================================================================

def softmax(x):
    """
    Softmax 函数（数值稳定版）。
    通过减去每行的最大值来防止指数运算溢出。
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)       # 减去最大值防溢出
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 当只有单个样本时，退化为向量
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    """
    交叉熵误差函数。
    :param y: 预测概率（经 Softmax 处理）
    :param t: 真实标签 (one-hot 或标签索引形式)
    :return: 平均损失值
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 如果 t 是 one-hot 编码，转换为标签索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # 为防止 log(0) 产生 -inf，加一个极小值 delta
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


class SoftmaxWithLoss:
    """
    将 Softmax 激活与交叉熵损失组合为一层。
    这样做的好处: 反向传播时可利用数学简化，得到简洁的梯度公式: y - t
    """

    def __init__(self):
        self.loss = None    # 损失值
        self.y = None       # Softmax 输出（概率）
        self.t = None       # 真实标签（one-hot 形式）

    def forward(self, x, t):
        """
        前向传播: 先 Softmax 再计算损失。
        :param x: 网络输出的 logits
        :param t: 真实标签
        :return: 标量损失值
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        """
        反向传播: 返回简化的梯度 (y - t) / batch_size。
        这是 Softmax + 交叉熵 组合的优雅性质。
        """
        batch_size = self.t.shape[0]

        # 将标签转换为 one-hot 形式
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


# ============================================================================
# 8. SimpleConvNet 类 — 完整的卷积神经网络
# ============================================================================

class SimpleConvNet:
    """
    用于手写数字识别的简化卷积神经网络。
    网络结构: Conv → ReLU → Pool → Affine → ReLU → Affine → SoftmaxWithLoss

    超参数配置:
      - 卷积层: 30 个 5×5 滤波器, stride=1, pad=0
      - 池化层: 2×2 最大值池化, stride=2
      - 隐藏层: 100 个神经元（全连接）
      - 输出层: 10 个神经元（全连接，对应 0~9 数字类别）

    输入: (N, 1, 28, 28) — MNIST 单通道 28×28 灰度图像
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        """
        初始化 SimpleConvNet。

        :param input_dim:        输入数据维度 (通道, 高, 宽)，MNIST 为 (1, 28, 28)
        :param conv_param:       卷积层超参数字典
            - filter_num:  滤波器数量（输出通道数），默认 30
            - filter_size: 滤波器大小（高=宽），默认 5
            - pad:         填充大小，默认 0
            - stride:      步幅，默认 1
        :param hidden_size:      隐藏层（全连接层）神经元数量，默认 100
        :param output_size:      输出层神经元数量（类别数），默认 10
        :param weight_init_std:  权重初始化标准差，默认 0.01
        """
        # 提取卷积超参数
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]  # 输入图像的高（假设正方形）

        # 计算卷积层输出尺寸（MNIST 28×28 → 24×24，因为 5×5 卷积核、无填充）
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        # 池化层输出尺寸（24×24 → 12×12，因为 2×2 池化、步幅 2）
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        # ---- 初始化权重参数 ----
        # W1: 卷积层权重，形状 (FN, C, FH, FW)
        # 使用正态分布随机初始化，乘以小的标准差以保证训练初期梯度稳定
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)  # 偏置初始化为 0

        # W2: 第1个全连接层权重，形状 (池化输出总元素数, 隐藏层神经元数)
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        # W3: 第2个全连接层权重，形状 (隐藏层神经元数, 输出类别数)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # ---- 按顺序组装各层 ----
        # 使用 OrderedDict 确保层的前向/反向传播按照定义的顺序执行
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        # 最后一层（Softmax + 损失）单独存储，因为它的用法与中间层不同
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """
        推理预测: 按顺序执行所有层的前向传播。
        :param x: 输入数据 (N, C, H, W)
        :return:  网络输出的 logits
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        计算损失函数值。
        :param x: 输入数据
        :param t: 真实标签
        :return:  标量损失值
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):
        """
        通过误差反向传播计算所有参数的梯度。
        :param x: 输入数据
        :param t: 真实标签
        :return: 梯度字典，键名与 self.params 对应
        """
        # 前向传播（含损失计算）
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.last_layer.backward(dout)

        # 按前向传播的逆序逐层反向传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 提取各层参数的梯度
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


# ============================================================================
# 9. 演示与测试
# ============================================================================

def demo_im2col():
    """
    演示 im2col 函数的使用效果。
    """
    print("=" * 60)
    print(">>> im2col 函数演示 <<<")

    # 示例1: 单张图片 (batch=1, channel=3, 7×7)
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(f"输入形状: {x1.shape} → im2col 展开后: {col1.shape}")
    # 预期: (1, 3, 7, 7) → 输出尺寸 = (7-5)/1+1 = 3, 所以 (1*3*3=9, 3*5*5=75)

    # 示例2: 批量数据 (batch=10, channel=3, 7×7)
    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(f"输入形状: {x2.shape} → im2col 展开后: {col2.shape}")
    # 预期: (10, 3, 7, 7) → (10*3*3=90, 3*5*5=75)


def demo_convolution():
    """
    演示卷积层的前向传播。
    """
    print("\n>>> 卷积层演示 <<<")
    # 创建卷积层: 10 个 3 通道 5×5 滤波器
    conv = Convolution(
        W=np.random.randn(10, 3, 5, 5) * 0.01,  # 小随机权重
        b=np.zeros(10),                           # 零偏置
        stride=1,
        pad=0
    )
    x = np.random.randn(2, 3, 28, 28)  # 2 张 3 通道 28×28 图像
    out = conv.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    # MNIST: (2, 3, 28, 28) → OH = (28-5)/1+1 = 24 → (2, 10, 24, 24)


def demo_pooling():
    """
    演示池化层的前向传播。
    """
    print("\n>>> 池化层演示 <<<")
    pool = Pooling(pool_h=2, pool_w=2, stride=2)
    x = np.random.randn(2, 10, 24, 24)  # 承接卷积层输出
    out = pool.forward(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    # (2, 10, 24, 24) → OH = (24-2)/2+1 = 12 → (2, 10, 12, 12)


def demo_simple_convnet():
    """
    演示完整 SimpleConvNet 的前向传播、损失计算与梯度求解。
    """
    print("\n>>> SimpleConvNet 完整演示 <<<")
    network = SimpleConvNet(
        input_dim=(1, 28, 28),
        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
        hidden_size=100,
        output_size=10
    )

    # 模拟一个 batch 的 MNIST 数据: 3 张 28×28 单通道图像
    x = np.random.rand(3, 1, 28, 28)
    t = np.array([1, 3, 7])  # 真实标签

    # 前向传播 → 损失
    loss = network.loss(x, t)
    print(f"输入形状: {x.shape}")
    print(f"计算得到的损失值: {loss:.4f}")

    # 反向传播 → 梯度
    grads = network.gradient(x, t)
    print("各层权重梯度的形状:")
    print(f"  W1 (卷积层滤波器) 梯度形状: {grads['W1'].shape}")
    print(f"  b1 (卷积层偏置)   梯度形状: {grads['b1'].shape}")
    print(f"  W2 (全连接层1)    梯度形状: {grads['W2'].shape}")
    print(f"  b2 (全连接层1)    梯度形状: {grads['b2'].shape}")
    print(f"  W3 (全连接层2)    梯度形状: {grads['W3'].shape}")
    print(f"  b3 (全连接层2)    梯度形状: {grads['b3'].shape}")


def main():
    """
    主函数：依次演示 CNN 各组件的实现。
    """
    print("=" * 60)
    print("8.3.1 CNN 基础组件底层实现演示")
    print("=" * 60)

    demo_im2col()
    demo_convolution()
    demo_pooling()
    demo_simple_convnet()

    print("\n所有 CNN 基础组件演示完成！")


if __name__ == '__main__':
    main()
