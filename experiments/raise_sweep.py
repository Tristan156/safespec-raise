"""收益边界扫描：上下文长度 L × 方法，实测加速比与收益判据 γ/β/μ*。

目标（用于论文"收益边界"一章）：
  1. 在多个上下文长度 L 上同会话实测 AR / 固定k / RAISE 的 TPOT 与加速比；
  2. 每个 L 上同步标定 t1(目标单token) / td(草稿单token) / tb(目标块k+1)，
     得到 γ=td/t1、β=tb/t1，以及翻正门槛 μ* = kγ + β；
  3. 输出「加速比-L 曲线」与「门槛-L 曲线」，定位正收益区间与翻转点。

纪律：
  * 不改动任何既有代码与结果；全部输出写入 results_sweep/。
  * 断点续跑：每个 (L, method, profile) 单元独立落盘。
  * 计时口径与 spec_engine / raise_engine 完全一致（wall-clock，不含 prefill）。
  * 所有数字来自实际运行，不做任何平滑或外推。

用法：
  python raise_sweep.py --lens 512,1024,2048,4096,8192 \
      --methods ar,fixed4,raise_fixed4,raise_adapt \
      --profiles gov_report,qmsum,multi_news,lcc,repobench-p \
      --nreq 16 --k 4 --wait-free-gb 22
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from safespec.spec_engine import (load_model, greedy_decode, speculative_decode,
                         _fwd, _crop_to, _argmax_token)
from safespec.raise_engine import raise_decode
from safespec.data_loader import build_profiles, load_longbench_from_dir, PROFILES

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TARGET_DEFAULT = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DRAFT_DEFAULT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "longbench", "data")
OUT_DIR = os.path.join(HERE, "results_sweep")
T_MAX_NEW = int(os.environ.get("SWEEP_MAX_NEW", "128"))


def prog(m):
    print("[%s] %s" % (time.strftime("%F %T"), m), flush=True)


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


def wait_free_gb(need_gb, timeout_h=12):
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
        prog("等显存 free=%dMiB < %dMiB" % (free, need_gb * 1024))
        time.sleep(60)
    return False


# ------------------------------------------------------------------ 标定
def timed_forward(model, input_ids, past, reps, base_len):
    """在固定 KV 长度 base_len 上重复前向，取中位数（秒）。含 cuda.synchronize。"""
    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    ts = []
    for _ in range(reps):
        _crop_to(past, base_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _logits, past = _fwd(model, input_ids, past)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    _crop_to(past, base_len)
    return statistics.median(ts)


@torch.no_grad()
def calibrate(model, draft, tok, prompt_ids, k, reps):
    """在指定 prompt 长度上标定 t1 / td / tb。"""
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
    t_logits, t_past = _fwd(model, ids, None)
    d_logits, d_past = _fwd(draft, ids, None)
    base_t = t_past.get_seq_length()
    base_d = d_past.get_seq_length()
    nxt = _argmax_token(t_logits)                      # [1]

    t1 = timed_forward(model, nxt, t_past, reps, base_t)
    td = timed_forward(draft, nxt, d_past, reps, base_d)
    blk = nxt.view(1, 1).repeat(1, k + 1)              # 目标块 k+1 输入
    tb = timed_forward(model, blk, t_past, reps, base_t)
    return dict(ctx=int(base_t), t1_ms=t1 * 1e3, td_ms=td * 1e3, tb_ms=tb * 1e3,
                gamma=(td / t1 if t1 else None), beta=(tb / t1 if t1 else None))


# ------------------------------------------------------------------ 各方法
def run_one(method, model, draft, tok, ids, k):
    if method == "ar":
        toks, secs = greedy_decode(model, tok, ids, T_MAX_NEW)
        return toks, secs, {"target_fwd": len(toks), "draft_fwd": 0, "rounds": len(toks)}
    if method.startswith("fixed"):
        kk = int(method[5:]) if method[5:].isdigit() else k
        toks, secs, rounds = speculative_decode(model, draft, tok, ids, kk, T_MAX_NEW)
        return toks, secs, {"rounds": len(rounds), "target_fwd": len(rounds),
                            "draft_fwd": sum(r[1] for r in rounds),
                            "accept_per_round": (sum(r[0] for r in rounds) / len(rounds)
                                                 if rounds else None)}
    if method == "raise_fixed4":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=k,
            adapt_k=False, use_early_stop=False, collect_margins=False)
        return toks, secs, st
    if method == "raise_adapt":
        toks, secs, rounds, st = raise_decode(
            model, draft, tok, ids, max_new=T_MAX_NEW, k_init=k,
            adapt_k=True, use_early_stop=False, collect_margins=False)
        return toks, secs, st
    raise ValueError("unknown method " + method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--lens", default="512,1024,2048,4096,8192")
    ap.add_argument("--methods", default="ar,fixed4,raise_fixed4,raise_adapt")
    ap.add_argument("--profiles", default="gov_report,qmsum,multi_news,lcc,repobench-p")
    ap.add_argument("--nreq", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=9, help="标定重复次数")
    ap.add_argument("--wait-free-gb", type=float, default=0)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    want = [p.strip() for p in args.profiles.split(",") if p.strip()]

    if not wait_free_gb(args.wait_free_gb):
        prog("FATAL 等显存超时")
        return 2

    dt = DTYPES[args.dtype]
    prog("加载模型 dtype=%s" % args.dtype)
    model, tok = load_model(args.target, dtype=dt)
    draft, _ = load_model(args.draft, dtype=dt)
    tasks = load_longbench_from_dir(DATA_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for L in lens:
        profs = build_profiles(tasks, tok, max_input=L)
        use = [p for p in PROFILES if p in profs and p in want]
        prog("=== L=%d 剖面=%s ===" % (L, use))
        for prof in use:
            reqs = profs[prof]["eval"][:args.nreq]
            if not reqs:
                continue
            real_len = statistics.mean(len(r) for r in reqs)

            # --- 标定（取该 L 下第一个剖面作为代表，结果记录 ctx 实测值） ---
            cb = None
            try:
                cb = calibrate(model, draft, tok, reqs[0], args.k, args.reps)
                cb["profile"] = prof
                cb["L"] = L
                cb["req_len_mean"] = real_len
                if cb["gamma"] is not None:
                    cb["mu_star"] = args.k * cb["gamma"] + cb["beta"]
                    cb["ub"] = (args.k + 1) / cb["mu_star"]
                prog("  标定 L=%d %s ctx=%d t1=%.2fms td=%.2fms tb=%.2fms "
                     "γ=%.4f β=%.4f μ*=%.3f 上限=%.3f×"
                     % (L, prof, cb["ctx"], cb["t1_ms"], cb["td_ms"], cb["tb_ms"],
                        cb["gamma"], cb["beta"], cb["mu_star"], cb["ub"]))
            except Exception as e:
                prog("  标定失败 %s: %r" % (prof, e))

            for method in methods:
                name = "L%d_%s_%s_%s" % (L, method, args.dtype, prof)
                if cell_done(name, len(reqs)):
                    prog("  SKIP %s" % name)
                    continue
                recs = []
                for i, ids in enumerate(reqs):
                    try:
                        toks, secs, st = run_one(method, model, draft, tok, ids, args.k)
                    except Exception as e:
                        prog("  ERR %s req%d: %r" % (name, i, e))
                        continue
                    ntok = max(1, len(toks))
                    recs.append(dict(req=i, method=method, L=L, profile=prof,
                                     dtype=args.dtype, prompt_len=len(ids),
                                     n_tokens=len(toks),
                                     seconds=round(secs, 5),
                                     tpot_ms=round(secs / ntok * 1e3, 4),
                                     target_fwd=st.get("target_fwd"),
                                     draft_fwd=st.get("draft_fwd"),
                                     rounds=st.get("rounds"),
                                     accept_per_round=st.get("accept_per_round"),
                                     avg_k=st.get("avg_k"),
                                     saved_fix_fwd=st.get("saved_fix_fwd"),
                                     early_stop=st.get("early_stop"),
                                     tokens=toks))
                write_cell(name, recs)
                if recs:
                    tp = sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs) * 1e3
                    prog("  DONE %s n=%d tpot=%.3fms" % (name, len(recs), tp))
                if cb is not None:
                    all_rows.append(dict(cb, method=method))

    with open(os.path.join(OUT_DIR, "calib_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=1)
    prog("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
