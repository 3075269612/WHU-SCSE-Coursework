# 实验一：词向量

使用 Jieba 对中文语料分词，以 Gensim 训练 Skip-gram Word2Vec，并完成词语相似度、近邻词、类比关系和 PCA 可视化。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\word2vec_experiment.py
```

若已存在分词结果，可添加 `--reuse-segmented`。完整训练会生成较大的模型、文本词向量和处理中间文件，这些内容已由 `.gitignore` 排除。

## 关键输出

- `outputs/results/experiment_results.txt`
- `outputs/figures/pca_word_vectors.png`
- [匿名实验报告](./docs/report/REPORT.md)

