"""
9.3.1 RNN 模型案例：姓氏分类
============================
对应第六次实验 RNN 文本分类案例。优先读取 data/names/*.txt 官方姓氏
数据集；若本地没有该数据集，则使用内置小型姓氏数据保持离线可运行。
"""

from io import open
import glob
import math
import os
import random
import string
import time
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# all_letters 是模型允许出现的字符表。姓氏会被转换成 one-hot 字符序列，
# 因此字符表长度 n_letters 就是每个时间步输入向量的维度。
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 教材原案例使用 PyTorch 官方 names 数据集，路径通常是 data/names/*.txt。
# 当前项目目录没有该数据集，为保证脚本在课程规定环境中可以独立运行，
# 这里准备一个小型内置数据集作为回退；如果本地存在 data/names，则优先使用本地数据。
BUILTIN_NAMES = {
    "Chinese": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao"],
    "English": ["Smith", "Jackson", "Taylor", "Brown", "Wilson", "Walker", "Young"],
    "French": ["Dubois", "Moreau", "Laurent", "Lefevre", "Rousseau", "Simon"],
    "German": ["Schmidt", "Fischer", "Weber", "Meyer", "Wagner", "Becker"],
    "Italian": ["Rossi", "Russo", "Ferrari", "Bianchi", "Romano", "Ricci"],
    "Japanese": ["Sato", "Suzuki", "Takahashi", "Tanaka", "Watanabe", "Ito"],
    "Russian": ["Ivanov", "Petrov", "Smirnov", "Dovesky", "Volkov", "Sokolov"],
    "Scottish": ["Campbell", "Stewart", "MacDonald", "Robertson", "Murray", "Reid"],
}


def findFiles(path):
    """查找匹配通配符的文件，和教材中的 findFiles 写法保持一致。"""
    return glob.glob(path)


def unicodeToAscii(s):
    """把带重音的 Unicode 姓氏规范化为 ASCII 字符。

    例如 Ślusàrski 会被拆成基础字母和附加音标，再去掉附加音标，
    得到 Slusarski。这样可以让输入字符落在 all_letters 定义的字符表内。
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


def readLines(filename):
    """读取一个类别文件中的所有姓氏，并逐行完成 ASCII 规范化。"""
    lines = open(filename, encoding="utf-8").read().strip().split("\n")
    return [unicodeToAscii(line) for line in lines]


def load_name_data():
    """加载姓氏分类数据。

    返回:
        category_lines: dict[str, list[str]]
            键是类别名，如 English、Russian；值是该类别下的姓氏列表。
        all_categories: list[str]
            所有类别名，类别在列表中的下标就是模型训练时使用的标签编号。
    """
    category_lines = {}
    all_categories = []
    files = findFiles(str(DATA_DIR / "names" / "*.txt"))

    if files:
        for filename in files:
            category = os.path.splitext(os.path.basename(filename))[0]
            all_categories.append(category)
            category_lines[category] = readLines(filename)
    else:
        category_lines = {key: [unicodeToAscii(value) for value in values] for key, values in BUILTIN_NAMES.items()}
        all_categories = sorted(category_lines)

    return category_lines, all_categories


def letterToIndex(letter):
    """把单个字符映射为字符表下标。"""
    return all_letters.find(letter)


def letterToTensor(letter):
    """把单个字符转换为 one-hot 张量，形状为 (1, n_letters)。"""
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    """把一个姓氏转换为 RNN 可逐时间步读取的三维张量。

    输出形状为 (line_length, 1, n_letters):
        line_length: 姓氏字符个数，也就是 RNN 的时间步数。
        1: batch_size，本案例每次只输入一个姓氏。
        n_letters: one-hot 字符向量维度。
    """
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        idx = letterToIndex(letter)
        if idx >= 0:
            tensor[li][0][idx] = 1
    return tensor


class RNN(nn.Module):
    """简单循环神经网络，用于姓氏分类任务。

    当前时间步的输入字符 input 和上一时间步隐藏状态 hidden 会分别经过线性层，
    两者相加后通过 tanh 得到新的隐藏状态；最后把隐藏状态映射到类别空间，
    再用 LogSoftmax 输出各类别的对数概率。
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # i2h: 输入字符 one-hot -> 隐藏状态贡献。
        self.i2h = nn.Linear(input_size, hidden_size)
        # h2h: 上一时间步隐藏状态 -> 当前隐藏状态贡献。
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # h2o: 最后一个时间步的隐藏状态 -> 类别 logits。
        self.h2o = nn.Linear(hidden_size, output_size)
        # NLLLoss 需要输入对数概率，因此这里使用 LogSoftmax。
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # RNN 的核心递推公式:
        # h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
        hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        # 每个姓氏从全零隐藏状态开始读取。
        return torch.zeros(1, self.hidden_size)


def categoryFromOutput(output, all_categories):
    """把模型输出的对数概率转换为类别名和类别下标。"""
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def randomChoice(values):
    """从列表中随机抽取一个元素。"""
    return values[random.randint(0, len(values) - 1)]


def randomTrainingExample(category_lines, all_categories):
    """随机生成一次训练样本。

    返回的 category_tensor 是监督标签，line_tensor 是输入的姓氏字符序列。
    """
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def train(rnn, criterion, category_tensor, line_tensor, learning_rate):
    """执行一次随机样本训练。

    训练过程与教材一致:
        1. 初始化隐藏状态。
        2. 按字符顺序把姓氏喂给 RNN。
        3. 只用最后一个时间步输出做类别预测。
        4. 计算 NLLLoss 并反向传播。
        5. 手动执行参数 = 参数 - 学习率 * 梯度。
    """
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # 这里没有使用 torch.optim，目的是贴近教材中手动更新参数的示例。
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    """把训练耗时格式化为 'Xm Ys'。"""
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def evaluate(rnn, line_tensor):
    """推理阶段只做前向传播，不记录梯度。"""
    hidden = rnn.initHidden()
    with torch.no_grad():
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
    return output


def predict(rnn, all_categories, input_line, n_predictions=3):
    """输出某个姓氏最可能的前 n_predictions 个类别。"""
    print("\n> %s" % input_line)
    output = evaluate(rnn, lineToTensor(input_line))
    topv, topi = output.topk(n_predictions, 1, True)
    for i in range(n_predictions):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        print("(%.2f) %s" % (value, all_categories[category_index]))


def save_loss_plot(losses, filename):
    """保存损失曲线。

    课程规定环境中 torch 与 matplotlib 同时导入时可能触发 OpenMP 运行时冲突。
    为保持环境干净，这里用 Pillow 直接绘制一张简单 PNG，而不依赖 matplotlib。
    """
    from PIL import Image, ImageDraw

    width, height = 760, 420
    margin = 50
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    draw.text((margin, 18), "RNN name classification loss", fill="black")
    draw.text((width // 2 - 30, height - 30), "records", fill="black")
    draw.text((8, height // 2), "loss", fill="black")

    if len(losses) > 1:
        min_loss = min(losses)
        max_loss = max(losses)
        span = max(max_loss - min_loss, 1e-8)
        points = []
        for i, loss in enumerate(losses):
            x = margin + i * (width - 2 * margin) / (len(losses) - 1)
            y = height - margin - (loss - min_loss) * (height - 2 * margin) / span
            points.append((x, y))
        draw.line(points, fill="#2f6f9f", width=3)
        draw.text((width - margin - 115, margin + 8), f"min={min_loss:.3f}", fill="black")
        draw.text((width - margin - 115, margin + 28), f"max={max_loss:.3f}", fill="black")

    image.save(filename)


if __name__ == "__main__":
    # 固定随机种子，保证每次训练日志和预测结果大体稳定。
    random.seed(0)
    torch.manual_seed(0)
    category_lines, all_categories = load_name_data()
    n_categories = len(all_categories)
    # 如果输出为空列表，说明当前目录没有 data/names/*.txt，脚本会使用内置小数据集。
    print(findFiles(str(DATA_DIR / "names" / "*.txt")))
    print(unicodeToAscii("Ślusàrski"))
    print("类别数量:", n_categories)

    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    # 先用 Albert 的第一个字符做一次前向传播，展示未训练模型的输出形状和概率含义。
    input_tensor = lineToTensor("Albert")
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input_tensor[0], hidden)
    print(output)
    print(categoryFromOutput(output, all_categories))

    criterion = nn.NLLLoss()
    learning_rate = 0.005
    n_iters = 5000
    print_every = 1000
    plot_every = 200
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, n_iters + 1):
        # 每次迭代随机抽一个类别和一个姓氏，属于随机梯度下降。
        category, line, category_tensor, line_tensor = randomTrainingExample(category_lines, all_categories)
        output, loss = train(rnn, criterion, category_tensor, line_tensor, learning_rate)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = "OK" if guess == category else "NO (%s)" % category
            print(
                "%d %d%% (%s) %.4f %s / %s %s"
                % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct)
            )

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    save_loss_plot(all_losses, OUTPUT_DIR / "rnn_name_loss.png")

    predict(rnn, all_categories, "Dovesky")
    predict(rnn, all_categories, "Jackson")
    predict(rnn, all_categories, "Satoshi")
    print("损失曲线已保存: output/rnn_name_loss.png")
