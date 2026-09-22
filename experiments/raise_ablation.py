"""机制消融实验 v3（逐请求轮转设计）：补齐 M2 与 M3 的单变量归因，并顺带扩样 §3.9 的精度诊断。

背景：
  v2（raise_fair_baseline.py）已确认 M1 单独贡献 +10.35%（10/10 剖面一致），M3 增量 +0.34%。
  但 M2（置信度门控 / 在线自适应深度）从未在公平设计下做过单变量消融 —— "三机制"里第三个是空的。

单变量链条（全部含 M1，只在被测机制上不同）：
  fair_fixed4  = M1                     （仅把修正 token 并入下一轮；非严格 EOS；固定 k=4）
  raise_adapt  = M1 + M2a               （M2a = 按滑动窗口接受率在线调整 k，不启用 margin 早停）
  raise_full   = M1 + M2a + M2b + M3    （M2b = margin 置信度早停；M3 = 严格 EOS）

  ⇒ M2a 效应 = fair_fixed4 → raise_adapt
  ⇒ M2b 效应 = raise_adapt → raise_full（含 M3，需与 v2 的 raise_fixed4 对照区分 M3）
  ⇒ 同时重跑 ar / fixed4 / fair_fixed4，既提供同会话锚点，也构成对 v2 主要结论的独立复现。

附带产出（零额外成本）：
  每个请求各方法的输出序列与同会话 AR 逐 token 比对 → 收集内容型分歧点，
  累计后将 §3.9 的精度诊断样本量由 24 提升到数十个。

设计纪律：逐请求轮转 + 多遍重复（沿用 v2），抵消共享 GPU 负载漂移。
不改动任何既有代码与结果：输出写入新目录 results_ablation/。
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
OUT_DIR = os.path.join(HERE, "results_ablation")
T_MAX_NEW = int(os.environ.get("ABL_MAX_NEW", "128"))
MARGIN_TAU = float(os.environ.get("RAISE_MARGIN_TAU", "0.8"))


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
    if method == "fair_fixed4":          # M1
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=False, use_early_stop=False, eos_strict=False, collect_margins=False)
    elif method == "raise_adapt":        # M1 + M2a（在线调 k）
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=True, use_early_stop=False, eos_strict=False, collect_margins=False)
    elif method == "raise_full":         # M1 + M2a + M2b + M3
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=True, use_early_stop=True, margin_tau=MARGIN_TAU,
            eos_strict=True, collect_margins=False)
    else:
        raise ValueError("unknown method " + method)
    st["fix_fwd"] = 0
    st["avg_k"] = round(st.get("avg_k", 0.0), 3)
    return toks, secs, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--methods", default="ar,fixed4,fair_fixed4,raise_adapt,raise_full")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
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
    prog("就绪：limit=%d repeats=%d 方法=%s" % (args.limit, args.repeats, methods))

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
            order = methods[rot:] + methods[:rot]
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
                            if len(toks) == len(r0):
                                d = next((i for i, (a, b) in enumerate(zip(toks, r0)) if a != b), None)
                                rec["div_pos"] = d
                                rec["ar_tok"] = (r0[d] if d is not None else None)
                                rec["method_tok"] = (toks[d] if d is not None else None)
                recs[m].append(rec)

        for m in methods:
            write_cell(cells[m], recs[m])
            n = len(recs[m])
            tot = sum(r["n_tokens"] for r in recs[m])
            tp = sum(r["seconds"] for r in recs[m]) / max(1, tot) * 1000
            prog("  DONE %s n=%d tpot=%.3fms" % (cells[m], n, tp))

        ars = [r["seconds"] / max(1, r["n_tokens"]) for r in recs["ar"]]
        base = statistics.median(ars)
        for m in methods:
            if m == "ar":
                continue
            v = [r["seconds"] / max(1, r["n_tokens"]) for r in recs[m]]
            mu = [r["accept_per_round"] for r in recs[m] if r.get("accept_per_round")]
            ak = [r["avg_k"] for r in recs[m] if r.get("avg_k")]
            prog("    %-11s %-12s sp_med=%.4f mu=%.3f avg_k=%.2f" %
                 (prof, m, base / statistics.median(v),
                  statistics.mean(mu) if mu else float("nan"),
                  statistics.mean(ak) if ak else float("nan")))

    # ---------------- 汇总 ----------------
    AGG = collections.defaultdict(lambda: dict(sec=0.0, tok=0, n=0, cons=0, cmpd=0,
                                               rounds=0, tfwd=0, fixfwd=0,
                                               mu=[], ak=[], es=0))
    perprof = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    DIV = []
    for f in glob.glob(os.path.join(OUT_DIR, "*_%s_%s_*.jsonl" % (args.dtype, args.split))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            a = AGG[r["method"]]
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
                a["mu"].append(r["accept_per_round"])
            if r.get("avg_k"):
                a["ak"].append(r["avg_k"])
            if r.get("early_stop"):
                a["es"] += r["early_stop"]
            if r.get("div_pos") is not None:
                DIV.append({"profile": r["profile"], "req": r["req"], "method": r["method"],
                            "d": r["div_pos"], "ar": r["ar_tok"], "mk": r["method_tok"]})
            pp = perprof[r["profile"]][r["method"]]
            pp[0] += r["seconds"] / max(1, r["n_tokens"])
            pp[1] += 1

    print("\n=== 机制消融 v3 汇总 (limit=%d repeats=%d) ===" % (args.limit, args.repeats))
    print("%-13s %6s %10s %9s %9s %8s %8s %7s %7s" %
          ("method", "n", "TPOT(ms)", "speedup", "target_f", "consist%", "mu", "avg_k", "estop"))
    base = AGG["ar"]["sec"] / max(1, AGG["ar"]["tok"])
    for m, a in sorted(AGG.items()):
        tp = a["sec"] / max(1, a["tok"]) * 1000
        sp = base / (a["sec"] / max(1, a["tok"])) if m != "ar" else 1.0
        cs = (a["cons"] / a["cmpd"] * 100) if a["cmpd"] else float("nan")
        mu = statistics.mean(a["mu"]) if a["mu"] else float("nan")
        ak = statistics.mean(a["ak"]) if a["ak"] else float("nan")
        print("%-13s %6d %10.3f %9.4f %9d %8s %7s %7s %7d" %
              (m, a["n"], tp, sp, a["tfwd"],
               ("%.2f" % cs) if cs == cs else "-",
               ("%.3f" % mu) if mu == mu else "-",
               ("%.2f" % ak) if ak == ak else "-", a["es"]))

    def eff(a, b):
        if a in AGG and b in AGG:
            ta = AGG[a]["sec"] / max(1, AGG[a]["tok"])
            tb = AGG[b]["sec"] / max(1, AGG[b]["tok"])
            return (ta / tb - 1) * 100
        return None

    m1 = eff("fixed4", "fair_fixed4")
    m2a = eff("fair_fixed4", "raise_adapt")
    m2b_m3 = eff("raise_adapt", "raise_full")
    print("\n=== 单变量效应 ===")
    print("M1  (fixed4 -> fair_fixed4)          : %s" % (("%+.2f%%" % m1) if m1 is not None else "-"))
    print("M2a (fair_fixed4 -> raise_adapt)     : %s" % (("%+.2f%%" % m2a) if m2a is not None else "-"))
    print("M2b+M3 (raise_adapt -> raise_full)   : %s" % (("%+.2f%%" % m2b_m3) if m2b_m3 is not None else "-"))

    print("\n=== 内容型分歧点（供 §3.9 精度诊断扩样） ===")
    print("累计分歧点样本数：%d" % len(DIV))
    if DIV:
        byp = collections.Counter(d["profile"] for d in DIV)
        print("按剖面：%s" % dict(byp))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"aggregate": {m: {k: v for k, v in a.items() if k not in ("mu", "ak")}
                             for m, a in AGG.items()},
               "mu": {m: (statistics.mean(a["mu"]) if a["mu"] else None) for m, a in AGG.items()},
               "avg_k": {m: (statistics.mean(a["ak"]) if a["ak"] else None) for m, a in AGG.items()},
               "effects": {"M1": m1, "M2a": m2a, "M2b_plus_M3": m2b_m3},
               "n_divergence_points": len(DIV)},
              open(os.path.join(OUT_DIR, "ablation_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump(DIV, open(os.path.join(OUT_DIR, "divergence_points.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved: results_ablation/ablation_summary.json, divergence_points.json")
    print("=== END ===")


if __name__ == "__main__":
    sys.exit(main())
