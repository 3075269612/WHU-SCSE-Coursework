from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
CODE_DIR = RESULTS_DIR / "generated_code"
RAW_RESPONSES_PATH = RESULTS_DIR / "qwen_responses.json"
COMPARISON_PATH = RESULTS_DIR / "parameter_comparison.csv"
SUMMARY_PATH = RESULTS_DIR / "experiment_results.txt"
MODEL = "qwen-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def build_client() -> OpenAI:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置 DASHSCOPE_API_KEY 环境变量。")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def request(
    client: OpenAI,
    *,
    task_id: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    content = choice.message.content or ""
    usage = response.usage
    return {
        "task_id": task_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response": content.strip(),
        "finish_reason": choice.finish_reason,
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def experiment_specs() -> list[dict[str, Any]]:
    news = (
        "中国科学家团队发布新型海洋预报系统，可将台风路径预测误差降低约15%，"
        "并计划在沿海城市开展试点。"
    )
    return [
        {
            "task_id": "summary_vague",
            "system_prompt": "你是中文文本助手。",
            "user_prompt": f"总结这条新闻：{news}",
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 120,
        },
        {
            "task_id": "summary_structured",
            "system_prompt": "你是严谨的新闻编辑，只依据给定标题写作，不补充未提供的事实。",
            "user_prompt": (
                "请把下面新闻标题改写为一段40至60字的摘要。必须包含主体、成果、"
                f"量化效果和后续计划，不使用标题党措辞：\n{news}"
            ),
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 120,
        },
        {
            "task_id": "concept_explanation",
            "system_prompt": "你是一名面向大学新生的科普教师，解释准确、避免不必要术语。",
            "user_prompt": (
                "用不超过180字解释“量子计算”。先给一个生活类比，再说明量子比特、"
                "叠加和测量，最后指出它并非在所有问题上都比普通计算机快。"
            ),
            "temperature": 0.4,
            "top_p": 0.85,
            "max_tokens": 240,
        },
        {
            "task_id": "sentiment_analysis",
            "system_prompt": "你是情感分析分类器，必须严格按指定 JSON 格式输出。",
            "user_prompt": (
                '分析评论“功能很强，但启动太慢，客服处理问题倒是很及时。”的情感。'
                '输出 JSON：{"label":"正面/中性/负面/混合","positive_evidence":[],'
                '"negative_evidence":[],"confidence":0到1}'
            ),
            "temperature": 0.1,
            "top_p": 0.7,
            "max_tokens": 160,
        },
        {
            "task_id": "translation",
            "system_prompt": "你是技术文档译者，保持术语一致，不增加解释。",
            "user_prompt": (
                "将下句翻译成简洁自然的中文：Large language models can follow "
                "instructions, but their outputs still require verification in high-stakes scenarios."
            ),
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 100,
        },
        {
            "task_id": "knowledge_qa",
            "system_prompt": "你是自然语言处理课程助教，回答应简洁且指出关键差异。",
            "user_prompt": "用三点比较大语言模型与传统小模型在训练方式、能力范围和部署成本上的区别。",
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 260,
        },
        {
            "task_id": "code_palindrome",
            "system_prompt": "你是 Python 代码生成器。只输出一个 markdown Python 代码块，不写额外说明。",
            "user_prompt": (
                "编写函数 is_palindrome(text: str) -> bool。忽略大小写、空格和标点，"
                "支持 Unicode 字母与数字。不得使用第三方库，只附带5个有代表性的 assert 测试。"
            ),
            "temperature": 0.1,
            "top_p": 0.7,
            "max_tokens": 420,
        },
        {
            "task_id": "code_average",
            "system_prompt": "你是 Python 代码生成器。只输出一个 markdown Python 代码块，不写额外说明。",
            "user_prompt": (
                "编写函数 mean(values: list[float]) -> float。空列表时抛出 ValueError，"
                "不得使用 statistics 或 numpy，并附带 assert 测试。"
            ),
            "temperature": 0.1,
            "top_p": 0.7,
            "max_tokens": 220,
        },
        {
            "task_id": "code_api_csv",
            "system_prompt": "你是 Python 代码生成器。只输出一个 markdown Python 代码块，不写额外说明。",
            "user_prompt": (
                "编写可直接运行的 Python 脚本：使用标准库 urllib.request 从 "
                "https://jsonplaceholder.typicode.com/posts 获取 JSON，将 id、userId、title "
                "保存为 posts.csv。要求超时10秒、检查 HTTP 状态、UTF-8 编码并捕获网络异常。"
            ),
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 420,
        },
    ]


def parameter_specs() -> list[dict[str, Any]]:
    prompt = (
        "以“雨后的校园”为主题写一段80至120字的中文描写，包含视觉和听觉细节，"
        "不要使用诗歌分行。"
    )
    return [
        {
            "task_id": "parameter_low",
            "system_prompt": "你是中文写作助手。",
            "user_prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.5,
            "max_tokens": 220,
        },
        {
            "task_id": "parameter_medium",
            "system_prompt": "你是中文写作助手。",
            "user_prompt": prompt,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 220,
        },
        {
            "task_id": "parameter_high",
            "system_prompt": "你是中文写作助手。",
            "user_prompt": prompt,
            "temperature": 1.2,
            "top_p": 0.95,
            "max_tokens": 220,
        },
    ]


def extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    opening = re.search(r"```(?:python)?\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    return (opening.group(1) if opening else text).strip()


def validate_generated_code(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    validations: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record["task_id"]
        if not task_id.startswith("code_"):
            continue
        code = extract_python(record["response"])
        path = CODE_DIR / f"{task_id}.py"
        path.write_text(code + "\n", encoding="utf-8")
        result: dict[str, Any] = {"path": str(path.relative_to(ROOT_DIR))}
        try:
            ast.parse(code)
            result["syntax_valid"] = True
        except SyntaxError as exc:
            result.update(syntax_valid=False, error=str(exc))
            validations[task_id] = result
            continue
        if task_id in {"code_palindrome", "code_average"}:
            completed = subprocess.run(
                [sys.executable, str(path)],
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            result.update(
                executed=True,
                returncode=completed.returncode,
                stdout=completed.stdout.strip(),
                stderr=completed.stderr.strip(),
            )
        else:
            result.update(
                executed=False,
                note="含外部网络请求，仅完成语法检查，避免在自动验证阶段访问第三方服务。",
            )
        validations[task_id] = result
    return validations


def char_metrics(text: str) -> dict[str, float]:
    chars = [char for char in text if not char.isspace()]
    counts = Counter(chars)
    unique_ratio = len(counts) / len(chars) if chars else 0.0
    repeated_ratio = (
        sum(count - 1 for count in counts.values() if count > 1) / len(chars)
        if chars
        else 0.0
    )
    return {
        "char_count": len(chars),
        "unique_char_ratio": round(unique_ratio, 4),
        "repeated_char_ratio": round(repeated_ratio, 4),
    }


def write_comparison(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if not record["task_id"].startswith("parameter_"):
            continue
        row = {
            "task_id": record["task_id"],
            "temperature": record["temperature"],
            "top_p": record["top_p"],
            "elapsed_seconds": record["elapsed_seconds"],
            "completion_tokens": record["completion_tokens"],
            **char_metrics(record["response"]),
            "response": record["response"].replace("\n", " "),
        }
        rows.append(row)
    with COMPARISON_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_comparison(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = ["低随机性", "中等随机性", "高随机性"]
    diversity = [row["unique_char_ratio"] for row in rows]
    lengths = [row["char_count"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, diversity, color=["#4C78A8", "#72B7B2", "#F58518"])
    axes[0].set_title("不同参数下的字符多样性")
    axes[0].set_ylabel("去重字符占比")
    axes[0].set_ylim(0, max(diversity) * 1.25)
    axes[1].bar(labels, lengths, color=["#4C78A8", "#72B7B2", "#F58518"])
    axes[1].set_title("不同参数下的输出长度")
    axes[1].set_ylabel("非空白字符数")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("temperature 与 top_p 参数对 Qwen 输出的影响")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "parameter_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    records: list[dict[str, Any]],
    validations: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    by_id = {record["task_id"]: record for record in records}
    lines = [
        "Qwen 系列模型应用实验结果",
        f"模型：{MODEL}",
        f"成功请求数：{len(records)}",
        "",
        "一、Prompt 优化对比",
        "[宽泛 Prompt]",
        by_id["summary_vague"]["response"],
        "",
        "[结构化 Prompt]",
        by_id["summary_structured"]["response"],
        "",
        "二、多场景 Prompt 输出",
    ]
    for task_id, label in [
        ("concept_explanation", "概念解释"),
        ("sentiment_analysis", "情感分析"),
        ("translation", "语言转换"),
        ("knowledge_qa", "知识问答"),
    ]:
        lines.extend([f"[{label}]", by_id[task_id]["response"], ""])
    lines.append("三、代码生成与验证")
    for task_id, result in validations.items():
        lines.append(f"{task_id}: {json.dumps(result, ensure_ascii=False)}")
    lines.extend(["", "四、temperature/top_p 参数对比"])
    for row in rows:
        lines.append(
            f"{row['task_id']}: temperature={row['temperature']}, top_p={row['top_p']}, "
            f"字符数={row['char_count']}, 去重字符占比={row['unique_char_ratio']}"
        )
        lines.append(row["response"])
        lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = build_client()
    records: list[dict[str, Any]] = []
    for spec in experiment_specs() + parameter_specs():
        print(f"Calling {spec['task_id']} ...", flush=True)
        records.append(request(client, **spec))
    validations = validate_generated_code(records)
    rows = write_comparison(records)
    plot_comparison(rows)
    payload = {
        "model": MODEL,
        "base_url": BASE_URL,
        "records": records,
        "code_validations": validations,
    }
    RAW_RESPONSES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(records, validations, rows)
    print(SUMMARY_PATH)


def repair_palindrome() -> None:
    if not RAW_RESPONSES_PATH.exists():
        raise FileNotFoundError("请先运行完整实验，再执行单项修复。")
    payload = json.loads(RAW_RESPONSES_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    spec = next(item for item in experiment_specs() if item["task_id"] == "code_palindrome")
    replacement = request(build_client(), **spec)
    for index, record in enumerate(records):
        if record["task_id"] == "code_palindrome":
            replacement["debug_note"] = (
                "首次生成因 max_tokens=260 被截断，调整为精简5个测试且 max_tokens=420 后重试。"
            )
            records[index] = replacement
            break
    validations = validate_generated_code(records)
    rows = write_comparison(records)
    plot_comparison(rows)
    payload["records"] = records
    payload["code_validations"] = validations
    RAW_RESPONSES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(records, validations, rows)
    print(SUMMARY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen 大模型应用实验")
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="只显示实验任务，不调用 API。",
    )
    parser.add_argument(
        "--repair-palindrome",
        action="store_true",
        help="只重试被截断的回文代码任务，并更新已有结果。",
    )
    args = parser.parse_args()
    if args.show_plan:
        specs = experiment_specs() + parameter_specs()
        print(json.dumps(specs, ensure_ascii=False, indent=2))
        return
    if args.repair_palindrome:
        repair_palindrome()
        return
    run()


if __name__ == "__main__":
    main()
