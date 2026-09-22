"""RAISE 对比实验聚合：results_raise/*.jsonl -> 汇总表（CSV + Markdown）。

指标与论文口径一致：
  * TPOT(ms)      : 解码耗时 / 生成 token 数（不含 prefill），按请求 token 数加权
  * 加速比         : 同剖面 AR 的 TPOT / 该方法 TPOT（>1 为加速）
  * 输出一致率     : 与同会话 AR 输出逐 token 完全一致的请求占比
  * target/draft 前向次数：开销归因（RAISE 应满足 target_fwd == rounds）
用法：python raise_analyze.py --dir results_raise --out .
"""
import argparse
import collections
import csv
import glob
import json
import os


def load(dirpath):
    rows = []
    for f in glob.glob(os.path.join(dirpath, "*.jsonl")):
        base = os.path.basename(f)[:-6]
        parts = base.split("_")
        if len(parts) < 4:
            continue
        method, dtype, split, profile = parts[0], parts[1], parts[2], "_".join(parts[3:])
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            r.setdefault("method", method)
            r.setdefault("dtype", dtype)
            r.setdefault("split", split)
            r.setdefault("profile", profile)
            rows.append(r)
    return rows


def agg(rows, keyf):
    out = collections.defaultdict(lambda: dict(n=0, tok=0, sec=0.0, cons=0, cons_n=0,
                                               tf=0, df=0, rounds=0, saved=0))
    for r in rows:
        a = out[keyf(r)]
        a["n"] += 1
        a["tok"] += r.get("n_tokens", 0)
        a["sec"] += r.get("seconds", 0.0)
        if r.get("consistent") is not None:
            a["cons_n"] += 1
            a["cons"] += 1 if r["consistent"] else 0
        a["tf"] += r.get("target_fwd") or 0
        a["df"] += r.get("draft_fwd") or 0
        a["rounds"] += r.get("rounds") or 0
        a["saved"] += r.get("saved_fix_fwd") or 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_raise")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    rows = load(args.dir)
    if not rows:
        print("no data in", args.dir)
        return 1

    by_method = agg(rows, lambda r: r["method"])
    # AR 基准 TPOT（按 token 加权，全剖面合并）
    ar = by_method.get("ar")
    ar_tpot = (ar["sec"] / ar["tok"] * 1000) if ar and ar["tok"] else None

    order = ["ar", "fixed4", "dynk", "raise_fixed4", "raise_adapt", "raise_full"]
    names = [m for m in order if m in by_method] + \
            [m for m in by_method if m not in order]

    lines = ["| 方法 | 请求数 | 生成 token | TPOT(ms) | 加速比 vs AR | 输出一致率 | target前向 | draft前向 | 每轮target前向 |",
             "|---|---|---|---|---|---|---|---|---|"]
    csv_rows = []
    for m in names:
        a = by_method[m]
        tpot = a["sec"] / a["tok"] * 1000 if a["tok"] else 0
        spd = (ar_tpot / tpot) if (ar_tpot and tpot) else 0
        cons = (a["cons"] / a["cons_n"] * 100) if a["cons_n"] else None
        per_round = (a["tf"] / a["rounds"]) if a["rounds"] else None
        lines.append("| %s | %d | %d | %.2f | %.3f× | %s | %d | %d | %s |" % (
            m, a["n"], a["tok"], tpot, spd,
            ("%.1f%%" % cons) if cons is not None else "-",
            a["tf"], a["df"],
            ("%.2f" % per_round) if per_round is not None else "-"))
        csv_rows.append(dict(method=m, n=a["n"], tokens=a["tok"], tpot_ms=round(tpot, 3),
                             speedup_vs_ar=round(spd, 4),
                             consistency_pct=(round(cons, 2) if cons is not None else ""),
                             target_fwd=a["tf"], draft_fwd=a["df"],
                             target_fwd_per_round=(round(per_round, 3) if per_round else ""),
                             saved_fix_fwd=a["saved"]))

    os.makedirs(args.out, exist_ok=True)
    md = os.path.join(args.out, "raise_compare_summary.md")
    csvp = os.path.join(args.out, "raise_compare_summary.csv")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# RAISE vs 基线（同会话、同数据、同协议）\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("说明：加速比 = 同会话 AR 的 TPOT / 该方法 TPOT；一致率 = 与 AR 输出逐 token 完全一致的请求占比。\n")
    with open(csvp, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # 逐剖面明细
    per_prof = agg(rows, lambda r: (r["profile"], r["method"]))
    prof_md = os.path.join(args.out, "raise_compare_by_profile.md")
    with open(prof_md, "w", encoding="utf-8") as f:
        f.write("# 逐剖面明细\n\n| 剖面 | 方法 | 请求 | TPOT(ms) | 一致率 |\n|---|---|---|---|---|\n")
        for (p, m), a in sorted(per_prof.items()):
            tpot = a["sec"] / a["tok"] * 1000 if a["tok"] else 0
            cons = (a["cons"] / a["cons_n"] * 100) if a["cons_n"] else None
            f.write("| %s | %s | %d | %.2f | %s |\n" % (
                p, m, a["n"], tpot,
                ("%.1f%%" % cons) if cons is not None else "-"))

    print("\n".join(lines))
    print("\n-> %s\n-> %s\n-> %s" % (md, csvp, prof_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
