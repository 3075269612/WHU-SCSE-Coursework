# 实验二：文本分类

在 THUCNews 四分类子集上实现 TextCNN 与 BiLSTM，并比较 batch size、词嵌入维度、隐藏层维度、学习率和 dropout 对分类效果的影响。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\text_classification_experiment.py --epochs 3 --device cpu
```

快速检查：

```powershell
.venv\Scripts\python.exe src\text_classification_experiment.py --epochs 1 --max-train-size 80 --device cpu
```

## 关键输出

- `outputs/results/metrics.csv`
- `outputs/results/training_history.csv`
- `outputs/figures/loss_*.png`
- [匿名实验报告](./docs/report/REPORT.md)

