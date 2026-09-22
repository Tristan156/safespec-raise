"""离线一致性核查：results_raise/*.jsonl 里已存 tokens 序列，逐请求与同会话 AR 比对。

为什么需要它：raise_compare.py 写入的 `consistent` 字段实测全为 None（ref 未回填），
导致汇总表一致率显示 0.0%——这是"无数据"，不是真实结论。本脚本用已落盘的
token 序列在 CPU 上重算，不需要 GPU，不触碰任何运行中的实验。

指标：
  equal_%       : 输出序列与同剖面同 req 的 AR 序列【完全逐 token 相同】的请求占比
  first_div_med : 不一致请求的首个分歧位置中位数（越大越接近一致）
  ar_self_%     : 自检项，AR 与自己比对必须 100%（验证比对链路本身没坏）

用法：python raise_consistency.py --dir results_raise --dtype fp16 --split eval --out .
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys


def load(dirpath):
    recs = []
    for f in glob.glob(os.path.join(dirpath, "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_raise")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    recs = [r for r in load(args.dir)
            if r.get("dtype") == args.dtype and r.get("split") == args.split]
    if not recs:
        print("no records", args.dir, args.dtype, args.split)
        return 1

    # 参考序列：AR 在同剖面同请求上的输出
    ref = {}
    for r in recs:
        if r.get("method") == "ar" and r.get("tokens"):
            ref[(r.get("profile"), r.get("req"))] = r["tokens"]

    agg = collections.defaultdict(lambda: dict(n=0, eq=0, div=[], miss=0,
                                               len_only=0, real_div=0, dl=[]))
    for r in recs:
        m = r.get("method")
        toks = r.get("tokens")
        key = (r.get("profile"), r.get("req"))
        t = ref.get(key)
        a = agg[m]
        if t is None or toks is None:
            a["miss"] += 1
            continue
        a["n"] += 1
        if toks == t:
            a["eq"] += 1
        else:
            mn = min(len(toks), len(t))
            if toks[:mn] == t[:mn]:
                # 内容逐 token 相同、仅长度（终止/EOS 截断）不同 ——「长度型」
                a["len_only"] += 1
                a["dl"].append(len(toks) - len(t))
            else:
                # 真正在中途分歧 ——「内容型」
                a["real_div"] += 1
                d = next((i for i, (x, y) in enumerate(zip(toks, t)) if x != y), mn)
                a["div"].append(d)

    order = ["ar", "fixed4", "dynk", "raise_fixed4", "raise_adapt", "raise_full"]
    names = [m for m in order if m in agg] + [m for m in agg if m not in order]

    lines = ["| 方法 | 可比请求 | 完全一致 | 一致率 | 长度型不一致 | 内容型不一致 | 内容分歧中位位置 |",
             "|---|---|---|---|---|---|---|"]
    csv_rows = []
    for m in names:
        a = agg[m]
        rate = (a["eq"] / a["n"] * 100) if a["n"] else None
        med = int(statistics.median(a["div"])) if a["div"] else "-"
        lines.append("| %s | %d | %d | %s | %d | %d | %s |" % (
            m, a["n"], a["eq"],
            ("%.1f%%" % rate) if rate is not None else "-",
            a["len_only"], a["real_div"], med))
        csv_rows.append(dict(method=m, comparable=a["n"], equal=a["eq"],
                             equal_pct=(round(rate, 2) if rate is not None else ""),
                             length_only=a["len_only"], content_diverged=a["real_div"],
                             content_div_median=med, missing_ref=a["miss"]))

    print("\n".join(lines))
    head = ["# RAISE 输出一致性核查（离线，基于已落盘 token 序列）", ""]
    note = ["", "注：一致率 = 与同会话 AR 输出逐 token 完全相同的请求占比；",
            "AR 行的一致率应恒为 100%（自检，用于确认比对链路正常）。", ""]

    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, "raise_consistency.md")
    csvp = os.path.join(args.out, "raise_consistency.csv")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(head + lines + note) + "\n")
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print("\n-> %s\n-> %s" % (md, csvp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
