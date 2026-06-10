from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import classification_metrics
from src.models import MLPClassifier
from src.utils import set_seed


def _to_tensor_dataset(x, y: np.ndarray) -> TensorDataset:
    """将 scipy 稀疏矩阵转为小规模实验可接受的 dense tensor。"""
    x_dense = x.toarray().astype("float32")
    y_array = y.astype("int64")
    return TensorDataset(torch.from_numpy(x_dense), torch.from_numpy(y_array))


def _evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict[str, float], np.ndarray]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(float(loss.item()) * len(batch_y))
            predictions = torch.argmax(logits, dim=1)
            y_true.extend(batch_y.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    avg_loss = float(sum(losses) / max(len(y_true), 1))
    return avg_loss, metrics, np.array(y_pred)


def train_mlp(
    x_train,
    y_train: np.ndarray,
    x_val,
    y_val: np.ndarray,
    *,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device_name: str = "cpu",
    track_history: bool = True,
    patience: int | None = None,
    class_weights: np.ndarray | None = None,
) -> dict:
    """训练 MLP，并返回训练历史、验证指标和验证集最优模型。"""
    set_seed(seed)
    device = torch.device(device_name)
    train_dataset = _to_tensor_dataset(x_train, y_train)
    val_dataset = _to_tensor_dataset(x_val, y_val)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=2, dropout=dropout).to(device)
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_val_metrics = {
        "accuracy": 0.0,
        "macro_f1": -1.0,
        "macro_precision": 0.0,
        "macro_recall": 0.0,
    }
    final_val_metrics = best_val_metrics.copy()
    final_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 主模型需要完整训练曲线；超参搜索也要用验证集最优点，但可跳过训练集评估。
        if track_history:
            train_loss, train_metrics, _ = _evaluate_model(model, train_loader, criterion, device)
            val_loss, val_metrics, _ = _evaluate_model(model, val_loader, criterion, device)
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["train_macro_f1"].append(train_metrics["macro_f1"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
        else:
            val_loss, val_metrics, _ = _evaluate_model(model, val_loader, criterion, device)

        final_val_loss = val_loss
        final_val_metrics = val_metrics
        if val_metrics["macro_f1"] > best_val_metrics["macro_f1"]:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = val_loss
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "val_metrics": {key: float(value) for key, value in best_val_metrics.items()},
        "best_val_metrics": {key: float(value) for key, value in best_val_metrics.items()},
        "final_val_metrics": {key: float(value) for key, value in final_val_metrics.items()},
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "epochs_ran": int(epoch),
        "stopped_early": bool(stopped_early),
        "used_class_weights": class_weights.tolist() if class_weights is not None else None,
        "device": str(device),
    }


def predict_mlp(model: nn.Module, x, batch_size: int = 256, device_name: str = "cpu") -> np.ndarray:
    """对任意稀疏 TF-IDF 特征矩阵进行预测。"""
    dummy_y = np.zeros(x.shape[0], dtype="int64")
    dataset = _to_tensor_dataset(x, dummy_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name)
    criterion = nn.CrossEntropyLoss()
    _, _, y_pred = _evaluate_model(model.to(device), loader, criterion, device)
    return y_pred
