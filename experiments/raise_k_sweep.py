"""k 消融实验 v5：固定推测深度 k 对 BEC 各量（μ、β、加速比）的影响。

动机：本文全部实验固定 k=4，但 BEC 式(10)(11) 中 β=t_b/t_1 与 k 有关（验证块长度 = k+1），
      且每轮成本 k·t_d+t_b 随 k 线性增长；因此“k 取多少”是判据内的一个显式决策变量，
      审稿人会问“k 是否最优、怎么选”。本实验给出 k∈{2,4,6,8} 下的实测曲线。

设计（逐请求轮转 + 2 遍重复，抵消共享 GPU 负载漂移）：
  五组：ar（锚点）/ raise_k2 / raise_k4 / raise_k6 / raise_k8
  全部为 M1（修正 token 并入下一轮）+ 非严格 EOS，**只变 k**，构成 k 的单变量扫描。
  同会话、同数据、同 dtype、同生成上限。

产出：k 对 μ、β、γ、每轮成本、端到端加速比的影响表，用于回答“k 如何选”并检验
      BEC 对 k 的预测（每轮成本随 k 上升、μ 随 k 次线性增长 → 存在最优 k）。

不改动任何既有代码与结果：输出写入新目录 results_ksweep/。
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

from safespec.spec_engine import load_model, greedy_decode                    # noqa: E402
from safespec.raise_engine import raise_decode                                 # noqa: E402
from safespec.data_loader import build_profiles, load_longbench_from_dir, PROFILES  # noqa: E402

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TARGET_DEFAULT = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DRAFT_DEFAULT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "longbench", "data")
OUT_DIR = os.path.join(HERE, "results_ksweep")
T_MAX_NEW = int(os.environ.get("KS_MAX_NEW", "128"))
KS = [2, 4, 6, 8]


def prog(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def cell_path(n):
    return os.path.join(OUT_DIR, n + ".jsonl")


def cell_count(n):
    p = cell_path(n)
    if not os.path.exists(p):
        return 0
    try:
        return sum(1 for l in open(p, encoding="utf-8") if l.strip())
    except Exception:
        return 0


def write_cell(n, recs):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = cell_path(n)
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


def wait_proc_gone(pat, timeout_h=12):
    """等另一个实验进程结束。

    【必须用 <name>.py 作为匹配模式】—— 本进程自己的命令行里带有 `--wait-proc <name>`，
    若直接用 <name> 去 pgrep，会命中观察者自身与 pgrep 的 sh -c 包装进程，
    计数永远不为 0，导致死等。加 `.py` 后缀 + [x] 正则技巧后只匹配真正的实验进程。
    """
    pat_re = "[%s]%s" % (pat[0], pat[1:].replace(".", r"\.") + r"\.py")
    deadline = time.time() + timeout_h * 3600
    last = -1
    while time.time() < deadline:
        n = int(os.popen("pgrep -cf '%s' || true" % pat_re).read().strip() or 0)
        if n != last:
            prog("等待前序任务 %s：当前存活 %d" % (pat, n))
            last = n
        if n == 0:
            prog("前序任务已结束，开始本实验")
            return True
        time.sleep(60)
    prog("等待前序任务超时，继续")
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
        return toks, secs, {"target_fwd": len(toks)}
    if method.startswith("raise_k"):
        k = int(method.split("k")[1])
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=k,
            adapt_k=False, use_early_stop=False, eos_strict=False, collect_margins=False)
        st["k"] = k
        return toks, secs, st
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--methods", default="ar," + ",".join("raise_k%d" % k for k in KS))
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--wait-free-gb", type=float, default=22)
    ap.add_argument("--wait-proc", default="raise_framework_cmp")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    want = list(PROFILES) if args.profiles == "all" else args.profiles.split(",")
    if args.wait_proc:
        wait_proc_gone(args.wait_proc)
    if not wait_free_gb(args.wait_free_gb):
        prog("FATAL 等待显存超时")
        return 2

    prog("加载模型 dtype=%s" % args.dtype)
    dt = DTYPES[args.dtype]
    model, tok = load_model(args.target, dtype=dt)
    draft, _ = load_model(args.draft, dtype=dt)
    tasks = load_longbench_from_dir(DATA_DIR)
    profiles = build_profiles(tasks, tok, max_input=8192)
    prog("就绪：limit=%d repeats=%d 方法=%s" % (args.limit, args.repeats, methods))

    for prof in [p for p in PROFILES if p in profiles and p in want]:
        reqs = profiles[prof][args.split]
        if args.limit:
            reqs = reqs[:args.limit]
        need = len(reqs) * args.repeats
        cells = {m: "%s_%s_%s_%s" % (m, args.dtype, args.split, prof) for m in methods}
        if all(cell_count(cells[m]) >= need for m in methods):
            prog("SKIP %s" % prof)
            continue
        recs = {m: [] for m in methods}
        ref = {}
        for idx in range(args.repeats * len(reqs)):
            ridx = idx % len(reqs)
            ids = reqs[ridx]
            rot = idx % len(methods)
            ut = gpu_busy_pct()
            for m in (methods[rot:] + methods[:rot]):
                toks, secs, st = run_one(m, model, draft, tok, ids)
                rec = {"method": m, "profile": prof, "req": ridx, "pass": idx // len(reqs),
                       "dtype": args.dtype, "split": args.split, "n_tokens": len(toks),
                       "seconds": secs, "tokens": toks, "gpu_util": ut}
                rec.update({kk: vv for kk, vv in st.items() if kk != "margins"})
                if m == "ar":
                    ref.setdefault(ridx, []).append(toks)
                else:
                    r0 = (ref.get(ridx) or [None])[0]
                    if r0:
                        rec["consistent"] = bool(toks == r0)
                recs[m].append(rec)
        for m in methods:
            write_cell(cells[m], recs[m])
            n = len(recs[m])
            tot = sum(r["n_tokens"] for r in recs[m])
            prog("  DONE %s n=%d tpot=%.3fms" % (cells[m], n,
                                                 sum(r["seconds"] for r in recs[m]) / max(1, tot) * 1000))
        ars = [r["seconds"] / max(1, r["n_tokens"]) for r in recs["ar"]]
        base = statistics.median(ars)
        for m in methods:
            if m == "ar":
                continue
            v = [r["seconds"] / max(1, r["n_tokens"]) for r in recs[m]]
            mu = [r["accept_per_round"] for r in recs[m] if r.get("accept_per_round")]
            rd = sum(r.get("rounds", 0) for r in recs[m])
            prog("    %-11s %-9s sp_med=%.4f mu=%.3f rounds=%d" %
                 (prof, m, base / statistics.median(v),
                  statistics.mean(mu) if mu else float("nan"), rd))

    # ---------------- 汇总 ----------------
    AGG = collections.defaultdict(lambda: dict(sec=0.0, tok=0, n=0, cons=0, cmpd=0, rounds=0))
    perprof = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    for f in glob.glob(os.path.join(OUT_DIR, "*_%s_%s_*.jsonl" % (args.dtype, args.split))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            a = AGG[r["method"]]
            a["sec"] += r["seconds"]
            a["tok"] += r["n_tokens"]
            a["n"] += 1
            a["rounds"] += r.get("rounds", 0)
            if r.get("consistent") is not None:
                a["cmpd"] += 1
                a["cons"] += 1 if r["consistent"] else 0
            pp = perprof[r["profile"]][r["method"]]
            pp[0] += r["seconds"] / max(1, r["n_tokens"])
            pp[1] += 1

    base = AGG["ar"]["sec"] / max(1, AGG["ar"]["tok"])
    print("\n=== k 消融 v5 汇总 (limit=%d repeats=%d) ===" % (args.limit, args.repeats))
    print("%-10s %6s %10s %9s %9s %8s %9s %9s" %
          ("method", "n", "TPOT(ms)", "speedup", "rounds", "tok/round", "consist%", "k"))
    rows = []
    for m in methods:
        if m not in AGG:
            continue
        a = AGG[m]
        tp = a["sec"] / max(1, a["tok"]) * 1000
        sp = base / (a["sec"] / max(1, a["tok"])) if m != "ar" else 1.0
        cpr = a["tok"] / max(1, a["rounds"]) if a["rounds"] else float("nan")
        cs = (a["cons"] / a["cmpd"] * 100) if a["cmpd"] else float("nan")
        kk = "-" if m == "ar" else m.split("k")[1]
        print("%-10s %6d %10.3f %9.4f %9d %9.3f %9s %9s" %
              (m, a["n"], tp, sp, a["rounds"], cpr,
               ("%.2f" % cs) if cs == cs else "-", kk))
        rows.append({"method": m, "n": a["n"], "tpot_ms": round(tp, 3), "speedup": round(sp, 4),
                     "rounds": a["rounds"], "tokens_per_round": (None if cpr != cpr else round(cpr, 3)),
                     "consist_pct": (None if cs != cs else round(cs, 2)), "k": kk})

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
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"rows": rows,
               "per_profile_speedup": {p: {m: (v[0] / v[1]) for m, v in perprof[p].items()}
                                       for p in perprof},
               "note": "只变 k；全部为 M1（修正 token 并入下一轮）+ 非严格 EOS"},
              open(os.path.join(OUT_DIR, "ksweep_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved: results_ksweep/ksweep_summary.json")
    print("=== END ===")


if __name__ == "__main__":
    sys.exit(main())
