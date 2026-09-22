"""RAISE vs 基线：同会话、同数据、同协议的横向对比实验。

设计原则（对齐实验纪律）：
  * 单变量：RAISE-fixed 与 fixed-k4 只差「增量回退 M1」；RAISE-adapt 再加「自适应深度 M2」。
  * 同源锚点：AR 在同一进程、同一数据上重跑，作为加速比分母与一致性基准。
  * 断点续跑：每个 (method, profile) 单元独立落盘，重跑自动跳过。
  * 不改动任何已有文件：全部输出写入 results_raise/。

用法：
  python raise_compare.py --dtype fp16 --split eval --profiles all \
      --methods ar,fixed4,dynk,raise_fixed4,raise_adapt --wait-free-gb 22
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)          # 项目根，供 spec_engine/data_loader

from safespec.spec_engine import (load_model, greedy_decode, speculative_decode,
                         dynamic_k_decode, seq_consistent)
from safespec.raise_engine import raise_decode
from safespec.data_loader import build_profiles, load_longbench_from_dir, PROFILES

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TARGET_DEFAULT = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DRAFT_DEFAULT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "longbench", "data")
OUT_DIR = os.path.join(HERE, "results_raise")
T_MAX_NEW = int(os.environ.get("RAISE_MAX_NEW", "128"))


def prog(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def cell_path(name):
    return os.path.join(OUT_DIR, name + ".jsonl")


def cell_done(name, n):
    p = cell_path(name)
    if not os.path.exists(p):
        return False
    try:
        with open(p, encoding="utf-8") as f:
            return sum(1 for l in f if l.strip()) >= n
    except Exception:
        return False


def write_cell(name, recs):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = cell_path(name)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def wait_free_gb(need_gb, timeout_h=24):
    """等 GPU 空闲显存达标再开跑，绝不杀他人进程。"""
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
            prog("显存就绪 free=%dMiB (需要 %dMiB)" % (free, need_gb * 1024))
            return True
        prog("等待显存：free=%dMiB < %dMiB，60s 后重试" % (free, need_gb * 1024))
        time.sleep(60)
    return False


# ------------------------------------------------------------------ 各方法
def run_one(method, model, draft, tok, ids):
    """返回 (tokens, seconds, stats)。所有方法计时口径一致（不含 prefill）。"""
    if method == "ar":
        toks, secs = greedy_decode(model, tok, ids, T_MAX_NEW)
        return toks, secs, {"target_fwd": len(toks), "draft_fwd": 0}
    if method == "fixed4":
        toks, secs, rounds = speculative_decode(model, draft, tok, ids, 4, T_MAX_NEW)
        return toks, secs, {"rounds": len(rounds),
                            "target_fwd": len(rounds),
                            "draft_fwd": sum(r[1] for r in rounds)}
    if method == "dynk":
        toks, secs = dynamic_k_decode(model, draft, tok, ids, T_MAX_NEW)
        return toks, secs, {}
    if method == "raise_fixed4":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=False, use_early_stop=False, collect_margins=False)
        return toks, secs, st
    if method == "raise_adapt":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=True, use_early_stop=False, collect_margins=False)
        return toks, secs, st
    if method == "raise_full":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=4,
            adapt_k=True, use_early_stop=True, margin_tau=float(
                os.environ.get("RAISE_MARGIN_TAU", "0.8")), collect_margins=False)
        return toks, secs, st
    raise ValueError("unknown method " + method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--split", default="eval", choices=["calib", "eval"])
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--methods", default="ar,fixed4,dynk,raise_fixed4,raise_adapt")
    ap.add_argument("--limit", type=int, default=0, help="每剖面请求数上限，0=全部")
    ap.add_argument("--wait-free-gb", type=float, default=0)
    ap.add_argument("--ref-method", default="ar",
                    help="一致性比对的参考方法（同一次运行内）")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    want = list(PROFILES) if args.profiles == "all" else args.profiles.split(",")

    if not wait_free_gb(args.wait_free_gb):
        prog("FATAL 等待显存超时，退出")
        return 2

    prog("加载模型 dtype=%s" % args.dtype)
    dt = DTYPES[args.dtype]
    model, tok = load_model(args.target, dtype=dt)
    draft, _ = load_model(args.draft, dtype=dt)
    tasks = load_longbench_from_dir(DATA_DIR)
    profiles = build_profiles(tasks, tok, max_input=8192)
    prog("就绪：%d 剖面，方法=%s" % (len([p for p in PROFILES if p in profiles]), methods))

    for prof in [p for p in PROFILES if p in profiles and p in want]:
        reqs = profiles[prof][args.split]
        if args.limit:
            reqs = reqs[:args.limit]
        # 参考方法的输出（用于一致性比对）
        ref_tokens = {}
        ref_path = cell_path("%s_%s_%s_%s" % (args.ref_method, args.dtype, args.split, prof))
        if os.path.exists(ref_path):
            with open(ref_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        ref_tokens[r["req"]] = r.get("tokens")

        for method in methods:
            name = "%s_%s_%s_%s" % (method, args.dtype, args.split, prof)
            if cell_done(name, len(reqs)):
                prog("SKIP %s (已完成)" % name)
                # 已完成单元也要保证参考序列可用（可能上次运行中断）
                if method == args.ref_method and not ref_tokens:
                    with open(cell_path(name), encoding="utf-8") as f:
                        ref_tokens = {json.loads(l)["req"]: json.loads(l).get("tokens")
                                      for l in f if l.strip()}
                continue
            recs = []
            for i, ids in enumerate(reqs):
                t0 = time.time()
                toks, secs, st = run_one(method, model, draft, tok, ids)
                wall = time.time() - t0
                ref = ref_tokens.get(i)
                rec = {
                    "req": i, "method": method, "dtype": args.dtype,
                    "split": args.split, "profile": prof,
                    "n_tokens": len(toks),
                    "seconds": round(secs, 4),
                    "tpot_ms": round(secs / max(1, len(toks)) * 1000, 3),
                    "wall_s": round(wall, 3),
                    "consistent": (seq_consistent(toks, ref) if ref is not None else None),
                    "target_fwd": st.get("target_fwd"),
                    "draft_fwd": st.get("draft_fwd"),
                    "rounds": st.get("rounds"),
                    "saved_fix_fwd": st.get("saved_fix_fwd"),
                    "avg_k": st.get("avg_k"),
                    "tokens": toks,
                }
                recs.append(rec)
                prog("  %s req%d tokens=%d tpot=%.2fms wall=%.1fs" %
                     (name, i, len(toks), rec["tpot_ms"], wall))
            write_cell(name, recs)
            # 关键修复：本方法若是参考方法（AR），立即回填，供后续方法做一致性比对。
            # 原实现只在剖面循环开始前读一次文件，导致同一次运行内 AR 先跑完时
            # 后续所有方法拿到 ref=None，consistent 字段恒为 None（汇总表显示 0.0% 假象）。
            if method == args.ref_method:
                ref_tokens = {r["req"]: r.get("tokens") for r in recs}
                prog("  已回填参考序列 %d 条，供后续方法一致性比对" % len(ref_tokens))
            prog("DONE %s (%d req)" % (name, len(recs)))

    prog("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
