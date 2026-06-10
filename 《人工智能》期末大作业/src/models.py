from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """用于 TF-IDF 文本特征的多层感知机分类器。"""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

