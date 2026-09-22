"""LongBench 派生的 10 个请求剖面构造。

任务 → 剖面映射（对齐原论文表 3）：
  代码续写: repobench-p, lcc
  摘要报告: multi_news, qmsum, samsum-long, gov_report
  长文问答: narrativeqa, multifieldqa, hotpotqa-long, qasper
每个剖面 = 任务来源 + 统一提示模板 + 输入长度上限 8192 + 后端设置。
"""
import os
import json
import glob

# 论文剖面名 -> instruction（简化、可复现的统一模板）
INSTRUCTIONS = {
    "repobench-p": "Please complete the following code.",
    "lcc": "Please complete the following code.",
    "multi_news": "Summarize the following news into a concise summary.",
    "qmsum": "Summarize the meeting into a concise summary.",
    "samsum-long": "Summarize the dialogue into a concise summary.",
    "gov_report": "Summarize the report into a concise summary.",
    "narrativeqa": "Answer the question based on the given document.",
    "multifieldqa": "Answer the question based on the given document.",
    "hotpotqa-long": "Answer the question based on the given document.",
    "qasper": "Answer the question based on the given paper.",
}

# 摘要类任务
_SUMM = ("multi_news", "qmsum", "samsum-long", "gov_report")
# 代码类任务
_CODE = ("repobench-p", "lcc")

PROFILES = {
    "repobench-p": "code",
    "lcc": "code",
    "multi_news": "summ",
    "qmsum": "summ",
    "samsum-long": "summ",
    "gov_report": "summ",
    "narrativeqa": "qa",
    "multifieldqa": "qa",
    "hotpotqa-long": "qa",
    "qasper": "qa",
}

# 论文剖面名 -> LongBench jsonl 文件名
FILENAME_ALIAS = {
    "repobench-p": "repobench-p",
    "lcc": "lcc",
    "multi_news": "multi_news",
    "qmsum": "qmsum",
    "samsum-long": "samsum",
    "gov_report": "gov_report",
    "narrativeqa": "narrativeqa",
    "multifieldqa": "multifieldqa_en",
    "hotpotqa-long": "hotpotqa",
    "qasper": "qasper",
}

# jsonl 文件名 -> 论文剖面名
_FILENAME_TO_PROFILE = {v: k for k, v in FILENAME_ALIAS.items()}

# 论文表 3 的校准/评估样本数
SPLIT = {
    "repobench-p": (30, 30), "lcc": (30, 30),
    "multi_news": (20, 20), "qmsum": (20, 20),
    "samsum-long": (20, 20), "gov_report": (20, 20),
    "narrativeqa": (20, 20), "multifieldqa": (20, 20),
    "hotpotqa-long": (20, 20), "qasper": (20, 20),
}


def build_prompt(task, sample):
    ctx = sample.get("context", "") or ""
    inp = sample.get("input", "") or ""
    if task in _CODE:
        # 代码续写：context=代码（repobench 含文件路径+代码），input=签名（repobench）或空（lcc）
        body = f"{ctx}\n{inp}".strip() if inp else ctx
        return f"Please complete the following code.\n\n{body}"
    if task in _SUMM:
        return f"Summarize the following text.\n\n{ctx}\n\nSummary:"
    # 问答类
    return f"Answer the question based on the following document.\n\n{ctx}\n\nQuestion: {inp}\nAnswer:"


def load_longbench_from_dir(data_dir):
    """从 LongBench 官方 data/*.jsonl 目录加载，返回 {论文剖面名: [samples]}。"""
    tasks = {}
    for f in glob.glob(os.path.join(data_dir, "*.jsonl")):
        fname = os.path.splitext(os.path.basename(f))[0]
        profile = _FILENAME_TO_PROFILE.get(fname)
        if profile is None:
            continue
        samples = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        tasks[profile] = samples
    return tasks


def build_profiles(tasks, tokenizer, max_input=8192):
    """把任务样本格式化为 (prompt_ids, profile, split)，按论文表3切分。"""
    profiles = {}
    for task, samples in tasks.items():
        n_cal, n_eval = SPLIT[task]
        need = n_cal + n_eval
        prompts = []
        for s in samples[:need]:
            p = build_prompt(task, s)
            ids = tokenizer(p, truncation=True, max_length=max_input)["input_ids"]
            prompts.append(ids)
        profiles[task] = {"calib": prompts[:n_cal], "eval": prompts[n_cal:n_eval + n_cal]}
    return profiles
