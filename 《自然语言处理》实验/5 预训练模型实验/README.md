# 实验五：预训练模型

使用 PyTorch 从零实现适合 CPU 运行的 Mini-BERT，同时训练掩码语言模型（MLM）和下一句预测（NSP），并比较网络深度与隐藏维度对参数量和收敛速度的影响。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\bert_pretraining_experiment.py
```

快速检查可使用 `--quick`，仅运行基线配置可使用 `--baseline-only`。

## 关键输出

- `outputs/results/parameter_comparison.csv`
- `outputs/results/sample_prediction.json`
- `outputs/figures/training_curves.png`
- [匿名实验报告](./docs/report/REPORT.md)

