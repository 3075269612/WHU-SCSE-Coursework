# 自然语言处理实验

本目录整理了六次自然语言处理课程实验，覆盖从经典词向量、文本分类和结构化预测，到预训练模型与大模型 API 应用的完整实践链路。公开内容仅保留实验代码、必要数据、结果摘要、可视化图表和匿名化 Markdown 报告。

## 实验目录

| 实验 | 主题 | 核心方法 | 报告 |
| --- | --- | --- | --- |
| 1 | 词向量 | Jieba、Word2Vec、PCA | [查看报告](./1%20词向量实验/docs/report/REPORT.md) |
| 2 | 文本分类 | TextCNN、BiLSTM、超参数扫描 | [查看报告](./2%20文本分类实验/docs/report/REPORT.md) |
| 3 | 句法分析 | Biaffine Dependency Parser | [查看报告](./3%20句法分析实验/docs/report/REPORT.md) |
| 4 | 实体识别 | CNN、BiLSTM、CRF、预训练词向量 | [查看报告](./4%20实体识别实验/docs/report/REPORT.md) |
| 5 | 预训练模型 | Mini-BERT、MLM、NSP | [查看报告](./5%20预训练模型实验/docs/report/REPORT.md) |
| 6 | 大模型应用 | Qwen API、Prompt 设计、代码验证 | [查看报告](./6%20大模型应用/docs/report/REPORT.md) |

## 目录约定

每个实验尽量采用相同结构：

```text
实验目录/
├── data/               # 原始数据；processed 为可重建中间数据
├── docs/
│   ├── assignment/     # 课程任务书
│   ├── reference/      # 教师提供或实验参考代码
│   └── report/         # 匿名 Markdown 报告；Word 报告仅本地保留
├── outputs/
│   ├── figures/        # 报告引用的图表
│   └── results/        # 指标、日志摘要和样例输出
├── src/                # 实验主程序
├── README.md
└── requirements.txt
```

模型权重、词向量、处理后数据、构建缓存和 Word 版个人报告不会上传到 GitHub，可通过实验脚本重新生成。

## 环境与运行

建议使用 Python 3.10，并优先通过 [uv](https://docs.astral.sh/uv/) 管理环境。进入某个实验目录后执行：

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe src\<实验脚本>.py
```

各实验的具体命令、参数和运行条件见对应目录下的 `README.md`。实验六需要自行设置 `DASHSCOPE_API_KEY`，密钥不得写入代码或提交到仓库。

## 说明

- 实验结果基于固定随机种子和当前保存配置，重跑时可能因软件版本或硬件差异出现轻微波动。
- 原始数据与参考代码仅用于课程学习，使用时应遵守其原始授权与数据许可。
- 仓库中的公开报告不包含姓名、学号、密钥或本机绝对路径。
