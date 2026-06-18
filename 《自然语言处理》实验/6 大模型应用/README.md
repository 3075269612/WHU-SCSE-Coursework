# 实验六：大模型应用

通过阿里云百炼的 OpenAI 兼容接口调用 Qwen，完成 Prompt 优化、多场景文本生成、代码生成与安全验证，并比较采样参数对输出的影响。

## 运行

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
$env:DASHSCOPE_API_KEY = "your-api-key"
.venv\Scripts\python.exe src\qwen_application_experiment.py
```

只查看调用计划而不访问 API：

```powershell
.venv\Scripts\python.exe src\qwen_application_experiment.py --show-plan
```

API 密钥只应通过环境变量提供，不得写入源码或提交到仓库。

## 关键输出

- `outputs/results/qwen_responses.json`
- `outputs/results/parameter_comparison.csv`
- `outputs/results/generated_code/`
- `outputs/figures/parameter_effects.png`
- [匿名实验报告](./docs/report/REPORT.md)

