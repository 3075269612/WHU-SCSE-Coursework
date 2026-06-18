# 实验三：依存句法分析

基于 THU 中文依存树库实现紧凑版 Biaffine 依存解析器，对比词向量维度、初始化方式和优化器，并使用 UAS、LAS 评价解析结果。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\dependency_parsing_experiment.py --epochs 5 --train-limit 1200 --dev-limit 400 --batch-size 16 --device cpu
```

快速检查：

```powershell
.venv\Scripts\python.exe src\dependency_parsing_experiment.py --epochs 1 --train-limit 40 --dev-limit 20 --batch-size 8 --device cpu
```

## 关键输出

- `outputs/results/metrics.csv`
- `outputs/results/sample_parse.txt`
- `outputs/figures/uas_las_comparison.png`
- [匿名实验报告](./docs/report/REPORT.md)

