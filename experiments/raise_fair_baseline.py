"""公平性对照实验 v2：逐请求轮转（interleaved）设计，消除方法执行顺序带来的负载偏差。

v1 的问题（已发现并废弃）：
  逐方法串行（先把 ar 全跑完、再跑 fixed4、再跑 fair_fixed4…）。在共享 GPU 上负载会漂移，
  先跑的方法系统性占便宜。v1 在 repobench-p 上已观测到 raise_fixed4 反而慢于 fixed4，
  与历史 +9.2% 矛盾 —— 该矛盾无法用"顺序漂移"或"真实效应"区分，故必须改设计。

v2 设计：
  * 逐请求轮转：for pass: for req: for method（方法顺序按 pass*req 轮转），
    使任一方法在序列中的位置均匀分布，负载漂移对所有方法等权。
  * 多次重复（--repeats），每剖面每方法得到 len(reqs)*repeats 个样本。
  * 同会话 AR 锚点；同数据、同协议、同 k=4、同 dtype。

四组（单变量隔离）：
  ar           自回归锚点
  fixed4       原基线（Option A：失配后单独为修正 token 补一次 target 前向 + 一次 draft 前向）
  fair_fixed4  M1 only（Option B：修正 token 并入下一轮 block），非严格 EOS
  raise_fixed4 M1 + M3（严格 EOS）

判读：fair_fixed4 相对 fixed4 的差值 = M1 在本文实现路径上消除的冗余开销占比。
不改动任何既有代码与结果：新增输出写入 results_fair/。
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from safespec.spec_engine import load_model, greedy_decode, speculative_decode   # noqa: E402
from safespec.raise_engine import raise_decode                                    # noqa: E402
from safespec.data_loader import build_profiles, load_longbench_from_dir, PROFILES  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TARGET_DEFAULT = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DRAFT_DEFAULT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "longbench", "data")
OUT_DIR = os.path.join(HERE, "results_fair")
T_MAX_NEW = int(os.environ.get("FAIR_MAX_NEW", "128"))


def prog(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def cell_path(name):
    return os.path.join(OUT_DIR, name + ".jsonl")


def cell_count(name):
    p = cell_path(name)
    if not os.path.exists(p):
        return 0
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip())
    except Exception:
        return 0


def write_cell(name, recs):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = cell_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def wait_free_gb(need_gb, timeout_h=24):
    if need_gb <= 0:
        return True
    deadline = time.time() + timeout_h * 3600
    while time.time() < deadline:
        out = os.popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").read()
        try:
            free = int(out.strip().split("\n")[0])
        except Exception:
            free = 0
        if free >= need_gb * 1024:
            prog("显存就绪 free=%dMiB" % free)
            return True
        time.sleep(60)
    return False


def gpu_busy_pct():
    out = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
    try:
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def run_one(method, model, draft, tok, ids):
    if method == "ar":
        toks, secs = greedy_decode(model, tok, ids, T_MAX_NEW)
        return toks, secs, {"target_fwd": len(toks), "draft_fwd": 0}
    if method == "fixed4":
        toks, secs, rounds = speculative_decode(model, draft, tok, ids, 4, T_MAX_NEW)
        return toks, secs, {"rounds": len(rounds), "target_fwd": len(rounds),
                            "draft_fwd": sum(r[1] for r in rounds),
                            "fix_fwd": sum(1 for r in rounds if r[0] < r[1])}
    if method == "fair_fixed4":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=False, use_early_stop=False, eos_strict=False,
            collect_margins=False)
        st["fix_fwd"] = 0
        return toks, secs, st
    if method == "raise_fixed4":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=False, use_early_stop=False, eos_strict=True,
            collect_margins=False)
        st["fix_fwd"] = 0
        return toks, secs, st
    raise ValueError("unknown method " + method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--methods", default="ar,fixed4,fair_fixed4,raise_fixed4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=2, help="每请求重复轮转的遍数")
    ap.add_argument("--wait-free-gb", type=float, default=22)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    want = list(PROFILES) if args.profiles == "all" else args.profiles.split(",")
    if not wait_free_gb(args.wait_free_gb):
        prog("FATAL 等待显存超时")
        return 2

    prog("加载模型 dtype=%s" % args.dtype)
    dt = DTYPES[args.dtype]
    model, tok = load_model(args.target, dtype=dt)
    draft, _ = load_model(args.draft, dtype=dt)
    tasks = load_longbench_from_dir(DATA_DIR)
    profiles = build_profiles(tasks, tok, max_input=8192)
    prog("就绪：repeats=%d 方法=%s" % (args.repeats, methods))

    for prof in [p for p in PROFILES if p in profiles and p in want]:
        reqs = profiles[prof][args.split]
        if args.limit:
            reqs = reqs[:args.limit]
        need = len(reqs) * args.repeats
        cells = {m: "%s_%s_%s_%s" % (m, args.dtype, args.split, prof) for m in methods}
        if all(cell_count(cells[m]) >= need for m in methods):
            prog("SKIP %s (已完成 %d)" % (prof, need))
            continue

        recs = {m: [] for m in methods}
        ref = {}
        for k in range(args.repeats * len(reqs)):
            ridx = k % len(reqs)
            ids = reqs[ridx]
            rot = k % len(methods)
            order = methods[rot:] + methods[:rot]      # 轮转方法顺序
            ut = gpu_busy_pct()
            for m in order:
                toks, secs, st = run_one(m, model, draft, tok, ids)
                rec = {"method": m, "profile": prof, "req": ridx, "pass": k // len(reqs),
                       "dtype": args.dtype, "split": args.split, "n_tokens": len(toks),
                       "seconds": secs, "tokens": toks, "gpu_util": ut}
                rec.update({kk: vv for kk, vv in st.items() if kk != "margins"})
                if m == "ar":
                    ref.setdefault(ridx, []).append(toks)
                else:
                    r0 = (ref.get(ridx) or [None])[0]
                    if r0:
                        rec["consistent"] = bool(toks == r0)
                        if toks != r0:
                            rec["len_type"] = (len(toks) != len(r0))
                            rec["content_type"] = (len(toks) == len(r0))
                recs[m].append(rec)

        for m in methods:
            write_cell(cells[m], recs[m])
            n = len(recs[m])
            tot = sum(r["n_tokens"] for r in recs[m])
            tp = sum(r["seconds"] for r in recs[m]) / max(1, tot) * 1000
            prog("  DONE %s n=%d tpot=%.3fms" % (cells[m], n, tp))

        # 剖面级小结（用剖面内 AR 锚点，中位数抗离群）
        ars = [r["seconds"] / max(1, r["n_tokens"]) for r in recs["ar"]]
        base = statistics.median(ars)
        for m in methods:
            if m == "ar":
                continue
            v = [r["seconds"] / max(1, r["n_tokens"]) for r in recs[m]]
            prog("    %-10s %-13s speedup_med=%.4f  speedup_mean=%.4f" %
                 (prof, m, base / statistics.median(v), base / statistics.mean(v)))

    # ---------------- 汇总 ----------------
    agg = collections.defaultdict(lambda: dict(sec=0.0, tok=0, n=0, cons=0, cmpd=0,
                                               rounds=0, tfwd=0, fixfwd=0, mu_sum=0.0, mu_n=0))
    perprof = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for f in glob.glob(os.path.join(OUT_DIR, "*_%s_%s_*.jsonl" % (args.dtype, args.split))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            a = agg[r["method"]]
            a["sec"] += r["seconds"]
            a["tok"] += r["n_tokens"]
            a["n"] += 1
            if r.get("consistent") is not None:
                a["cmpd"] += 1
                a["cons"] += 1 if r["consistent"] else 0
            if r.get("rounds"):
                a["rounds"] += r["rounds"]
            if r.get("target_fwd"):
                a["tfwd"] += r["target_fwd"]
            if r.get("fix_fwd"):
                a["fixfwd"] += r["fix_fwd"]
            if r.get("accept_per_round"):
                a["mu_sum"] += r["accept_per_round"]
                a["mu_n"] += 1
            pp = perprof[r["profile"]][r["method"]]
            pp[0] += r["seconds"] / max(1, r["n_tokens"])
            pp[1] += 1

    print("\n=== 公平性对照 v2 汇总 (dtype=%s split=%s repeats=%d) ===" % (args.dtype, args.split, args.repeats))
    print("%-14s %7s %11s %10s %9s %8s %9s %7s" %
          ("method", "req", "TPOT(ms)", "speedup", "target_f", "fix_fwd", "consist%", "mu"))
    base = None
    if agg.get("ar"):
        base = agg["ar"]["sec"] / max(1, agg["ar"]["tok"])
    for m, a in sorted(agg.items()):
        tp = a["sec"] / max(1, a["tok"]) * 1000
        sp = (base / (a["sec"] / max(1, a["tok"]))) if (m != "ar" and base) else 1.0
        cs = (a["cons"] / a["cmpd"] * 100) if a["cmpd"] else float("nan")
        mu = (a["mu_sum"] / a["mu_n"]) if a["mu_n"] else float("nan")
        print("%-14s %7d %11.3f %10.4f %9d %8d %9.2f %7.3f" %
              (m, a["n"], tp, sp, a["tfwd"], a["fixfwd"], cs, mu))

    print("\n=== 逐剖面 speedup（相对该剖面 AR 均值） ===")
    print("%-14s" % "profile" + "".join("%12s" % m for m in methods))
    for p in sorted(perprof):
        bm = perprof[p].get("ar")
        b = (bm[0] / bm[1]) if bm else None
        row = "%-14s" % p
        for m in methods:
            v = perprof[p].get(m)
            row += "%12s" % (("%.4f" % (b / (v[0] / v[1]))) if (v and b and m != "ar")
                             else ("1.0000" if m == "ar" else "-"))
        print(row)
    print("=== END ===")


if __name__ == "__main__":
    sys.exit(main())
