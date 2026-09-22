"""框架层对照实验 v4：与 HuggingFace transformers 官方推测解码实现同条件对比。

动机（回应审稿意见「缺少与其他相关文献/主流框架方法比较」）：
  v2/v3 的对照都是自研实现内部的机制拆解，尚未与「主流推理库的公开实现」直接比较。
  本实验补上这一点：transformers（5.15.0）自带两套公开推测解码实现——
    A) assisted_generation：model.generate(assistant_model=draft, do_sample=False)
       —— 即 Leviathan 式草稿-验证，由官方实现，是当前最主流的公开参考实现之一；
    B) prompt_lookup_decoding：prompt_lookup_num_tokens=k
       —— 免草稿模型的 n-gram 检索式推测解码，官方内置。
  二者都是公开、可复现、非本文自研的对照物。

附带目的：
  本实验同时回答一个此前悬而未决的问题——本文原基线 fixed4 在失配轮多付一次目标前向，
  这与官方实现是否一致？若官方实现同样如此，则该项冗余并非本文实现特有；
  若官方实现规避了它，则本文基线偏慢，RAISE 的增益需按实现路径限定（论文已按此口径写）。

计时口径（三方法必须一致才可比）：
  * 本文自研方法：沿用 spec_engine / raise_engine 的内部计时，天然不含 prefill。
  * HF 方法：TPOT = (T_generate - T_prefill) / n_tokens，其中 T_prefill 为对同一 prompt 的一次
    原始前向（use_cache=True）耗时。该口径在本脚本内统一施加，并在汇总中标注。
  同一请求的所有方法共用同一 prompt 与生成上限，逐请求轮转以抵消负载漂移。

不改动任何既有代码与结果：输出写入新目录 results_framework/。
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
OUT_DIR = os.environ.get("FW_OUT_DIR") or os.path.join(HERE, "results_framework")
T_MAX_NEW = int(os.environ.get("FW_MAX_NEW", "128"))
LOOKUP_N = int(os.environ.get("FW_LOOKUP_N", "4"))
CONF_TH = float(os.environ.get("FW_CONF_TH", "0.4"))    # HF 官方置信度门控阈值
K_MAX = int(os.environ.get("FW_K_MAX", "4"))            # 官方动态深度的初始/上限深度，与本文 k=4 对齐


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


def wait_proc_gone(pat, timeout_h=12):
    """等另一个实验进程结束，避免两个计时任务互相干扰。

    【必须用 <name>.py 作为匹配模式】—— 本进程自己的命令行里带有 `--wait-proc <name>`，
    若直接用 <name> 去 pgrep，会命中观察者自身（以及 pgrep 的 sh -c 包装进程），
    计数永远不为 0，导致死等。加 `.py` 后缀后只匹配真正的实验进程。
    同时屏蔽 pgrep 自身的自匹配（用 [x] 正则技巧）。
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


def hf_generate(model, tok, ids, kind, draft):
    """HF 官方实现；均强制生成 T_MAX_NEW 个 token 以与自研方法对齐。

    注：目标模型与草稿模型各自加载的 tokenizer 对象不同，transformers 会直接
    抛 "The main and assistant models have different tokenizers"，因此必须显式
    传入 tokenizer / assistant_tokenizer（两者同属 Qwen2.5，共享同一实例即可）。
    """
    kw = dict(do_sample=False, max_new_tokens=T_MAX_NEW, min_new_tokens=T_MAX_NEW, use_cache=True)
    if kind in ("hf_assisted", "hf_conf"):
        kw["assistant_model"] = draft
        kw["tokenizer"] = tok
        kw["assistant_tokenizer"] = tok
        if kind == "hf_conf":
            # transformers 官方提供的动态推测深度：按助手模型的置信度提前截断本轮候选，
            # 与本文的 M2a（置信度门控）构成同层对照。
            kw["assistant_confidence_threshold"] = CONF_TH
            kw["num_assistant_tokens"] = K_MAX
            kw["num_assistant_tokens_schedule"] = "heuristic"
    elif kind == "hf_lookup":
        kw["prompt_lookup_num_tokens"] = LOOKUP_N
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = model.generate(ids, **kw)
    except TypeError:
        # 该 transformers 版本不接受 tokenizer 关键字时，退回只传 assistant_tokenizer
        kw.pop("tokenizer", None)
        out = model.generate(ids, **kw)
    torch.cuda.synchronize()
    sec = time.time() - t0
    return out[0, ids.shape[-1]:].tolist(), sec


def run_one(method, model, draft, tok, ids, ids_t, prefill_s):
    if method == "ar":
        toks, secs = greedy_decode(model, tok, ids, T_MAX_NEW)
        return toks, secs, {"target_fwd": len(toks), "draft_fwd": 0}, "internal"
    if method == "fixed4":
        toks, secs, rounds = speculative_decode(model, draft, tok, ids, 4, T_MAX_NEW)
        return toks, secs, {"rounds": len(rounds), "target_fwd": len(rounds),
                            "fix_fwd": sum(1 for r in rounds if r[0] < r[1])}, "internal"
    if method == "fair_fixed4":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=False, use_early_stop=False, eos_strict=False, collect_margins=False)
        st["fix_fwd"] = 0
        return toks, secs, st, "internal"
    if method in ("hf_assisted", "hf_lookup", "hf_conf"):
        toks, secs_raw = hf_generate(model, tok, ids_t, method, draft)
        secs = max(0.0, secs_raw - prefill_s)      # 扣除 prefill，与自研口径对齐
        return toks, secs, {"hf_raw_sec": round(secs_raw, 4), "prefill_sec": round(prefill_s, 4)}, "hf_minus_prefill"
    raise ValueError("unknown method " + method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--methods", default="ar,fixed4,fair_fixed4,hf_assisted,hf_lookup")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--wait-free-gb", type=float, default=22)
    ap.add_argument("--wait-proc", default="raise_ablation")
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
            prog("SKIP %s (已完成 %d)" % (prof, need))
            continue

        recs = {m: [] for m in methods}
        ref = {}
        for k in range(args.repeats * len(reqs)):
            ridx = k % len(reqs)
            ids = reqs[ridx]
            ids_t = torch.tensor([ids], device=model.device, dtype=torch.long)
            # prefill 探针（每请求每遍测一次，供 HF 方法扣除）
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                model(ids_t, use_cache=True)
            torch.cuda.synchronize()
            prefill_s = time.time() - t0
            rot = k % len(methods)
            order = methods[rot:] + methods[:rot]
            ut = gpu_busy_pct()
            for m in order:
                toks, secs, st, how = run_one(m, model, draft, tok, ids, ids_t, prefill_s)
                rec = {"method": m, "profile": prof, "req": ridx, "pass": k // len(reqs),
                       "dtype": args.dtype, "split": args.split, "n_tokens": len(toks),
                       "seconds": secs, "tokens": toks, "gpu_util": ut, "timing": how}
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

        ars = [r["seconds"] / max(1, r["n_tokens"]) for r in recs["ar"]]
        base = statistics.median(ars)
        for m in methods:
            if m == "ar":
                continue
            v = [r["seconds"] / max(1, r["n_tokens"]) for r in recs[m]]
            cs = [r for r in recs[m] if r.get("consistent") is not None]
            rate = (sum(1 for r in cs if r["consistent"]) / len(cs) * 100) if cs else float("nan")
            prog("    %-11s %-12s sp_med=%.4f consist=%.1f%%" %
                 (prof, m, base / statistics.median(v), rate))

    # ---------------- 汇总 ----------------
    AGG = collections.defaultdict(lambda: dict(sec=0.0, tok=0, n=0, cons=0, cmpd=0,
                                               tfwd=0, fixfwd=0))
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
            if r.get("consistent") is not None:
                a["cmpd"] += 1
                a["cons"] += 1 if r["consistent"] else 0
            if r.get("target_fwd"):
                a["tfwd"] += r["target_fwd"]
            if r.get("fix_fwd"):
                a["fixfwd"] += r["fix_fwd"]
            pp = perprof[r["profile"]][r["method"]]
            pp[0] += r["seconds"] / max(1, r["n_tokens"])
            pp[1] += 1

    print("\n=== 框架层对照 v4 汇总 (limit=%d repeats=%d) ===" % (args.limit, args.repeats))
    print("%-14s %6s %10s %9s %9s %8s %9s" %
          ("method", "n", "TPOT(ms)", "speedup", "target_f", "fix_fwd", "consist%"))
    base = AGG["ar"]["sec"] / max(1, AGG["ar"]["tok"])
    for m, a in sorted(AGG.items()):
        tp = a["sec"] / max(1, a["tok"]) * 1000
        sp = base / (a["sec"] / max(1, a["tok"])) if m != "ar" else 1.0
        cs = (a["cons"] / a["cmpd"] * 100) if a["cmpd"] else float("nan")
        print("%-14s %6d %10.3f %9.4f %9d %8d %9s" %
              (m, a["n"], tp, sp, a["tfwd"], a["fixfwd"],
               ("%.2f" % cs) if cs == cs else "-"))

    print("\n=== 逐剖面 speedup（相对该剖面 AR 均值） ===")
    print("%-14s" % "profile" + "".join("%14s" % m for m in methods))
    for p in sorted(perprof):
        bm = perprof[p].get("ar")
        b = (bm[0] / bm[1]) if bm else None
        row = "%-14s" % p
        for m in methods:
            v = perprof[p].get(m)
            row += "%14s" % (("%.4f" % (b / (v[0] / v[1]))) if (v and b and m != "ar")
                             else ("1.0000" if m == "ar" else "-"))
        print(row)

    # 一致性：HF 官方实现 vs 本文实现
    print("\n=== 输出一致性（相对同会话自回归） ===")
    for m in methods:
        a = AGG.get(m)
        if not a or not a["cmpd"]:
            continue
        print("  %-14s %d/%d = %.2f%%" % (m, a["cons"], a["cmpd"], a["cons"] / a["cmpd"] * 100))

    json.dump({"aggregate": {m: dict(n=a["n"],
                                     tpot_ms=round(a["sec"] / max(1, a["tok"]) * 1000, 3),
                                     speedup=round(base / (a["sec"] / max(1, a["tok"])), 4)
                                     if m != "ar" else 1.0,
                                     target_fwd=a["tfwd"], fix_fwd=a["fixfwd"],
                                     consist_pct=(round(a["cons"] / a["cmpd"] * 100, 2) if a["cmpd"] else None))
                                 for m, a in AGG.items()},
               "per_profile_speedup": {p: {m: (v[0] / v[1]) for m, v in perprof[p].items()}
                                       for p in perprof},
               "timing_note": "自研方法为内部计时（不含 prefill）；HF 方法为 (T_generate - T_prefill)/n_tokens"},
              open(os.path.join(OUT_DIR, "framework_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("saved: results_framework/framework_summary.json")
    print("=== END ===")


if __name__ == "__main__":
    sys.exit(main())
