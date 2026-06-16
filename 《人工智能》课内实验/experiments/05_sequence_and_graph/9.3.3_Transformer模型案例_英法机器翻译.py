"""
9.3.3 Transformer 模型案例：英法机器翻译
=======================================
对应第六次实验 Transformer seq2seq 案例。优先读取 fra-eng/fra.txt
官方 Tatoeba 英法平行语料；若本地没有该文件，则使用内置小语料保持离线可运行。
"""

import math
import os
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# 教材使用 Tatoeba 英法平行语料 fra-eng/fra.txt。
# 为保证本课程目录在离线状态下也能运行，若本地没有 fra-eng/fra.txt，
# 则使用下面的小型平行语料。后续流程仍然保持“读数据 -> 预处理 -> 词表 ->
# padding -> Transformer 编码器/解码器 -> 训练 -> 预测 -> BLEU”的教材结构。
MINI_FRA_ENG = """Go.\tVa !
Hi.\tSalut !
Run!\tCours !
Run!\tCourez !
Who?\tQui ?
Wow!\tCa alors !
I lost.\tJ'ai perdu .
He's calm.\tIl est calme .
I'm home.\tJe suis chez moi .
"""


def read_data_nmt():
    """载入英语-法语数据集。"""
    # 教材默认数据位置是 fra-eng/fra.txt，本项目未强制下载该数据集。
    data_file = PROJECT_ROOT / "fra-eng" / "fra.txt"
    if data_file.exists():
        # 若存在真实数据集，优先读取真实数据，代码无需修改即可切换到完整语料。
        with open(data_file, "r", encoding="utf-8") as f:
            return f.read()
    # 若不存在真实数据集，则使用内置小语料，保证完整 Transformer 流程仍可执行。
    return MINI_FRA_ENG


def preprocess_nmt(text):
    """预处理英语-法语数据集。

    1. 替换特殊空白字符。
    2. 统一小写，降低词表规模。
    3. 在标点符号前插入空格，使标点作为独立 token。
    """

    def no_space(char, prev_char):
        # 只有当前字符是标点，且它前面没有空格时，才需要额外插入空格。
        return char in set(",.!?") and prev_char != " "

    # \u202f 和 \xa0 是语料中常见的非标准空白字符，先统一成普通空格。
    text = text.replace("\u202f", " ").replace("\xa0", " ").lower()
    # 逐字符扫描，在英文/法文单词与标点之间插入空格，方便后续 split 分词。
    out = [" " + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return "".join(out)


def tokenize_nmt(text, num_examples=None):
    """将制表符分隔的双语句对拆成源语言和目标语言 token 列表。"""
    source, target = [], []
    for i, line in enumerate(text.splitlines()):
        # num_examples 用于截取前若干条样本，便于课堂实验快速运行。
        if num_examples is not None and i >= num_examples:
            break
        # manythings/Tatoeba 数据通常是 “英文\t法文\t其他信息”，这里只取前两列。
        parts = line.split("\t")
        if len(parts) >= 2:
            # 本实验使用最朴素的空格分词，标点已在 preprocess_nmt 中变成独立 token。
            source.append(parts[0].split())
            target.append(parts[1].split())
    return source, target


class Vocab:
    """机器翻译词表，支持 token 到索引的转换。"""

    def __init__(self, tokens, min_freq=1, reserved_tokens=None):
        reserved_tokens = reserved_tokens or []
        # 统计所有句子中 token 的出现频次。
        counter = Counter(token for line in tokens for token in line)
        # 下标 0 固定给 <unk>，后面依次放 <pad>、<bos>、<eos> 等保留 token。
        self.idx_to_token = ["<unk>"] + reserved_tokens
        # 按词频从高到低加入词表；频次低于 min_freq 的 token 会被丢弃。
        for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            if freq >= min_freq and token not in self.idx_to_token:
                self.idx_to_token.append(token)
        # 反向索引表，后续可把 token 快速转成整数编号。
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        # 如果传入的是 token 列表，则递归地逐个 token 转换为编号列表。
        if isinstance(tokens, list):
            return [self.__getitem__(token) for token in tokens]
        # 词表外 token 统一映射到 <unk>，避免推理时因为生词报错。
        return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])

    def to_tokens(self, indices):
        # 支持单个下标和下标列表两种输入，便于解码预测结果。
        if isinstance(indices, int):
            return self.idx_to_token[indices]
        return [self.idx_to_token[int(index)] for index in indices]


def truncate_pad(line, num_steps, padding_token):
    """将序列截断或填充到固定长度。"""
    if len(line) > num_steps:
        # 超长序列只保留前 num_steps 个 token，保证一个 batch 内张量长度一致。
        return line[:num_steps]
    # 短序列在末尾补 <pad>，不会影响有效长度内的训练损失。
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将 token 序列转为小批量张量，并计算有效长度。"""
    # 每个样本末尾追加 <eos>，模型预测到该 token 时停止生成。
    lines = [vocab[line] + [vocab["<eos>"]] for line in lines]
    # 统一截断/补齐到 num_steps，得到形状为 (num_examples, num_steps) 的整数张量。
    array = torch.tensor([truncate_pad(line, num_steps, vocab["<pad>"]) for line in lines])
    # 有效长度不统计 <pad>，后续用于 attention mask 和 loss mask。
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回数据迭代器、源语言词表、目标语言词表和原始 token。"""
    # 先读取并预处理原始文本，再拆分成源语言/目标语言两份 token 序列。
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    # 源语言和目标语言使用独立词表，因为两种语言的 token 空间不同。
    src_vocab = Vocab(source, min_freq=1, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = Vocab(target, min_freq=1, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    # 把 token 序列转换为整数矩阵和有效长度。
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # TensorDataset 的每条样本包含: 源序列、源有效长度、目标序列、目标有效长度。
    dataset = TensorDataset(src_array, src_valid_len, tgt_array, tgt_valid_len)
    # shuffle=True 对训练更友好；小语料中主要用于复现完整训练流程。
    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_iter, src_vocab, tgt_vocab, source, target


def sequence_mask(X, valid_len, value=0):
    """按有效长度遮蔽序列。

    X 的形状通常是 (batch_size, num_steps)，valid_len 表示每行真实长度。
    超过真实长度的 padding 位置会被填成 value。
    """
    # maxlen 是序列长度，例如 num_steps。
    maxlen = X.size(1)
    # mask 的形状为 (batch_size, maxlen)，有效位置为 True，padding 位置为 False。
    mask = torch.arange(maxlen, device=X.device)[None, :] < valid_len[:, None]
    # 把 padding 位置替换成指定 value。用于 loss 时 value=0，用于 attention 时 value=-1e6。
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """带 mask 的 softmax，用于忽略 padding 位置。"""
    if valid_lens is None:
        # 没有有效长度时，直接对最后一维做普通 softmax。
        return nn.functional.softmax(X, dim=-1)
    shape = X.shape
    if valid_lens.dim() == 1:
        # X 通常是 (batch, query_len, key_len)，每个 query 共享同一个 key 有效长度。
        valid_lens = torch.repeat_interleave(valid_lens, shape[1])
    else:
        # 解码器训练时 valid_lens 可能是 (batch, query_len)，需要展平成一维。
        valid_lens = valid_lens.reshape(-1)
    # 将 padding 对应的 attention score 置为极小值，softmax 后权重接近 0。
    X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力: softmax(QK^T / sqrt(d))V。"""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (batch, query_len, d)
        # keys:    (batch, key_len, d)
        # values:  (batch, value_len, value_dim)，其中 key_len 通常等于 value_len。
        d = queries.shape[-1]
        # torch.bmm 执行批量矩阵乘法，得到每个 query 对每个 key 的相关性分数。
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # 对 key 维度做 masked softmax，得到注意力权重。
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 用注意力权重对 values 加权求和，得到上下文向量。
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_qkv(X, num_heads):
    """为多头注意力拆分头。

    输入形状: (batch_size, num_steps, num_hiddens)
    输出形状: (batch_size * num_heads, num_steps, num_hiddens / num_heads)
    """
    # 先把 hidden 维拆成 (num_heads, head_dim)。
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # 交换维度，让 head 维提前，便于把每个头当成独立 batch 计算注意力。
    X = X.permute(0, 2, 1, 3)
    # 合并 batch 和 head 维，后续 DotProductAttention 无需知道多头细节。
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """将多头注意力的多个头重新拼接。"""
    # 先还原为 (batch, num_heads, num_steps, head_dim)。
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # 调整回 (batch, num_steps, num_heads, head_dim)。
    X = X.permute(0, 2, 1, 3)
    # 最后拼接所有 head，得到 (batch, num_steps, num_hiddens)。
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力层。

    每个头在不同子空间内计算注意力，最后把多个头的输出拼接并线性变换。
    """

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        # 分别把 query/key/value 投影到 num_hiddens 维，再拆成多个头。
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        # 多头拼接后的输出再经过一次线性变换，回到模型隐藏维度。
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 线性投影 + 拆分多头。
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # 每个头都需要同一份有效长度，因此按 num_heads 重复。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        # 在每个头上独立计算缩放点积注意力。
        output = self.attention(queries, keys, values, valid_lens)
        # 将多个头重新拼接。
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络: Linear -> ReLU -> Linear。"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        # 第一个全连接层升维，增加非线性表达能力。
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        # 第二个全连接层降回 num_hiddens，保证可与残差连接相加。
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # FFN 对每个位置独立处理，不混合不同时间步的信息。
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """残差连接 + 层归一化。"""

    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # LayerNorm 在最后一个维度上归一化，即对每个 token 的隐藏向量归一化。
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # Y 是子层输出；X + Y 是残差连接；LayerNorm 稳定训练。
        return self.ln(self.dropout(Y) + X)


class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码。"""

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # P 预先保存 max_len 个位置的位置编码，形状为 (1, max_len, num_hiddens)。
        self.P = torch.zeros((1, max_len, num_hiddens))
        # X 的每一行对应一个位置，每一列对应一组不同频率。
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        # 偶数维使用 sin，奇数维使用 cos，这是原始 Transformer 的位置编码公式。
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X[:, : self.P[:, :, 1::2].shape[-1]])

    def forward(self, X):
        # X 是词嵌入；加上同长度的位置编码，让模型感知 token 顺序。
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


class EncoderBlock(nn.Module):
    """Transformer 编码器块: 自注意力 + AddNorm + FFN + AddNorm。"""

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        use_bias=False,
    ):
        super().__init__()
        # 编码器自注意力: Q、K、V 全都来自编码器输入 X。
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 前馈网络负责对每个位置的表示做进一步非线性变换。
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 第一个子层: 多头自注意力 + 残差归一化。
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 第二个子层: FFN + 残差归一化。
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """Transformer 编码器。"""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        use_bias=False,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        # token id -> 词向量。词向量维度等于模型隐藏维度 num_hiddens。
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        # 堆叠 num_layers 个编码器块。
        self.blks = nn.ModuleList(
            [
                EncoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    use_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.attention_weights = [None] * num_layers

    def forward(self, X, valid_lens):
        # 词嵌入乘 sqrt(num_hiddens) 是 Transformer 常用缩放，避免位置编码相对过大。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            # 逐层编码，每层输出作为下一层输入。
            X = blk(X, valid_lens)
            # 保存注意力权重，便于后续可视化或调试。
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


class DecoderBlock(nn.Module):
    """Transformer 解码器块。

    包含:
        1. 掩码自注意力，防止当前位置看到未来 token。
        2. 编码器-解码器注意力，对源语言编码结果做查询。
        3. 逐位置前馈网络。
    """

    def __init__(
        self,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        dropout,
        i,
    ):
        super().__init__()
        self.i = i
        # attention1: 解码器内部的掩码自注意力。
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # attention2: 编码器-解码器注意力，K/V 来自编码器输出。
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state[0]: 编码器输出；state[1]: 源序列有效长度；state[2]: 各解码层缓存。
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            # 训练时第一次进入该层，当前 X 就是 key/value。
            key_values = X
        else:
            # 预测时每次只输入一个 token，需要把历史已解码 token 缓存起来。
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = X.shape
            # 训练阶段一次性输入整个目标序列，用 [1,2,...,num_steps] mask 未来位置。
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 预测阶段逐 token 解码，不存在“未来 token”被同时输入的问题。
            dec_valid_lens = None

        # 掩码自注意力: 当前目标 token 只能看见自己和之前的目标 token。
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 编码器-解码器注意力: 目标端查询源语言编码表示。
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        # 前馈网络子层。
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    """Transformer 解码器。"""

    def __init__(
        self,
        vocab_size,
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        # 解码器也需要词嵌入和位置编码，输入是目标语言 token。
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        # 堆叠 num_layers 个解码器块，每个块都有独立缓存位置 i。
        self.blks = nn.ModuleList(
            [
                DecoderBlock(
                    key_size,
                    query_size,
                    value_size,
                    num_hiddens,
                    norm_shape,
                    ffn_num_input,
                    ffn_num_hiddens,
                    num_heads,
                    dropout,
                    i,
                )
                for i in range(num_layers)
            ]
        )
        # 将隐藏状态映射到目标语言词表大小，输出每个 token 的 logits。
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self._attention_weights = [[None] * num_layers for _ in range(2)]

    def init_state(self, enc_outputs, enc_valid_lens):
        # 第三个元素是每层解码器的历史 key/value 缓存，预测时会逐步填充。
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        # 和编码器一样，目标 token 嵌入也要乘 sqrt(num_hiddens) 再加位置编码。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            # 逐层解码，并维护 state 缓存。
            X, state = blk(X, state)
            # 保存两类注意力权重: 解码器自注意力、编码器-解码器注意力。
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights


class EncoderDecoder(nn.Module):
    """编码器-解码器整体模型。"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        # 先编码源语言序列，得到每个源 token 的上下文表示。
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        # 用编码器输出初始化解码器状态。
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        # 解码器根据目标端输入 dec_X 预测下一个 token。
        return self.decoder(dec_X, dec_state)


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """遮蔽 padding 位置的交叉熵损失。"""

    def forward(self, pred, label, valid_len):
        # 初始权重全为 1，之后 padding 位置会被 sequence_mask 改为 0。
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        # pred 原形状为 (batch, num_steps, vocab_size)，CrossEntropyLoss 需要类别维在第 2 维。
        self.reduction = "none"
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # 只统计非 padding 位置的损失，避免模型被填充 token 干扰。
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def grad_clipping(net, theta):
    """梯度裁剪，缓解序列模型训练中的梯度爆炸。"""
    params = [p for p in net.parameters() if p.requires_grad]
    # 计算所有参数梯度的整体 L2 范数。
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params if p.grad is not None))
    if norm > theta:
        # 如果梯度范数超过阈值，按比例缩小所有梯度。
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练 seq2seq Transformer。"""

    def xavier_init_weights(m):
        # 线性层用 Xavier 初始化，利于深层网络稳定训练。
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.GRU):
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    # 初始化参数并把模型移动到 CPU/GPU。
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    losses = []
    for epoch in range(1, num_epochs + 1):
        net.train()
        metric_loss, metric_tokens = 0.0, 0.0
        for batch in data_iter:
            optimizer.zero_grad()
            # X/Y 是源/目标序列；X_valid_len/Y_valid_len 是非 padding 长度。
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 教师强制: 解码器输入为 <bos> + 真实目标序列去掉最后一个 token。
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], dim=1)
            # 前向传播，Y_hat 形状为 (batch, num_steps, tgt_vocab_size)。
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            # l 是每条样本的损失，求和后反向传播。
            l.sum().backward()
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            # 记录总损失和 token 数，最后报告平均 token 损失。
            metric_loss += l.sum().item()
            metric_tokens += num_tokens.item()
        avg_loss = metric_loss / metric_tokens
        losses.append(avg_loss)
        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:03d}, loss {avg_loss:.4f}")
    return losses


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    """逐 token 贪心解码。"""
    net.eval()
    # 源句子预处理、分词、转编号，并在末尾加 <eos>。
    src_tokens = src_vocab[preprocess_nmt(src_sentence).split()] + [src_vocab["<eos>"]]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # 源序列也需要补齐到 num_steps，以匹配训练时的输入形状。
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])
    enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device).unsqueeze(0)
    # 编码源语言，初始化解码器状态。
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 解码从 <bos> 开始。
    dec_X = torch.tensor([[tgt_vocab["<bos>"]]], dtype=torch.long, device=device)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # 每轮只预测下一个 token。
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 贪心策略: 直接取概率最大的 token 作为当前输出。
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 预测到 <eos> 时停止，不把 <eos> 加入最终翻译。
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)
    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k=2):
    """计算 BLEU 分数。"""
    pred_tokens = pred_seq.split()
    label_tokens = label_seq.split()
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    if len_pred == 0:
        return 0.0
    # brevity penalty: 预测句子过短时进行惩罚。
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        # 统计参考译文中所有 n-gram 的出现次数。
        label_subs = Counter(tuple(label_tokens[i : i + n]) for i in range(len_label - n + 1))
        for i in range(len_pred - n + 1):
            ngram = tuple(pred_tokens[i : i + n])
            if label_subs[ngram] > 0:
                # 匹配到一个 n-gram 后减少计数，避免重复刷分。
                num_matches += 1
                label_subs[ngram] -= 1
        # 按 BLEU 的几何平均思想累计 n-gram 精度。
        score *= (num_matches / max(len_pred - n + 1, 1) + 1e-12) ** (0.5**n)
    return score


def save_loss_plot(losses, filename):
    """用 Pillow 保存损失曲线，避免 torch 与 matplotlib 的 OpenMP 运行时冲突。"""
    from PIL import Image, ImageDraw

    width, height = 760, 420
    margin = 50
    # 创建白底画布，后续用折线表示训练损失变化。
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    draw.text((margin, 18), "Transformer seq2seq loss", fill="black")
    draw.text((width // 2 - 25, height - 30), "epoch", fill="black")
    draw.text((8, height // 2), "loss", fill="black")
    if len(losses) > 1:
        min_loss, max_loss = min(losses), max(losses)
        span = max(max_loss - min_loss, 1e-8)
        points = []
        for i, loss in enumerate(losses):
            # 将 epoch 下标和 loss 映射到图片坐标系。
            x = margin + i * (width - 2 * margin) / (len(losses) - 1)
            y = height - margin - (loss - min_loss) * (height - 2 * margin) / span
            points.append((x, y))
        draw.line(points, fill="#2f6f9f", width=3)
        draw.text((width - margin - 130, margin + 8), f"final={losses[-1]:.4f}", fill="black")
    image.save(filename)


if __name__ == "__main__":
    # 固定随机种子，减少每次运行训练结果的波动。
    torch.manual_seed(0)
    # 展示原始语料片段和分词结果，便于对照教材前几步输出。
    raw_text = read_data_nmt()
    print(raw_text[:75])
    text = preprocess_nmt(raw_text)
    source_preview, target_preview = tokenize_nmt(text, num_examples=6)
    print((source_preview, target_preview))

    # 为了在课堂实验环境快速跑通，这里用小 batch 和小模型。
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 4, 10
    lr, num_epochs = 0.01, 200
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    # 自动选择 GPU；没有 GPU 时使用 CPU，本项目当前验证是在 CPU 上完成的。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建 DataLoader 和两套词表。
    train_iter, src_vocab, tgt_vocab, source, target = load_data_nmt(batch_size, num_steps)
    first_batch = next(iter(train_iter))
    print("源语言词表大小:", len(src_vocab), "目标语言词表大小:", len(tgt_vocab))
    print("第一个批次形状:", [tuple(x.shape) for x in first_batch])

    # 按教材流程分别创建编码器、解码器，再组合成 EncoderDecoder。
    encoder = TransformerEncoder(
        len(src_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    decoder = TransformerDecoder(
        len(tgt_vocab),
        key_size,
        query_size,
        value_size,
        num_hiddens,
        norm_shape,
        ffn_num_input,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
    )
    net = EncoderDecoder(encoder, decoder)

    # 训练模型，并保存每个 epoch 的平均 token 损失。
    losses = train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
    save_loss_plot(losses, OUTPUT_DIR / "transformer_loss.png")

    # 使用教材里的四条测试句，输出翻译和 BLEU 分数。
    engs = ["go .", "i lost .", "he's calm .", "i'm home ."]
    fras = ["va !", "j'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=True
        )
        print(f"{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}")

    print("训练损失曲线已保存: output/transformer_loss.png")
