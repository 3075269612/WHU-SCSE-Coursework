# 实验四：命名实体识别

在 ResumeNER 中文简历数据集上评估 CNN、BiLSTM、CRF 与预训练词向量的组合效果。该实验复用课程提供的 checkpoint，重点完成统一测试与实体级指标分析。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\ner_experiment.py
```

教师 checkpoint 位于 `outputs/models/`，是本实验评估流程的必要输入，因此予以保留。

## 关键输出

- `outputs/results/dataset_stats.json`
- `outputs/results/ner_metrics.csv`
- `outputs/figures/ner_model_f1_comparison.png`
- [匿名实验报告](./docs/report/REPORT.md)

