"""标定长上下文推测解码的收益条件：μ > kγ + β。

推导（贪心验证、固定 k）：
  每轮成本 C = k·t_d + t_b + p·t_b        （p = 该轮发生失配的概率，第 3 项即 W1 浪费）
  每轮产出 μ（= 平均接受 token 数）
  推测解码优于自回归  <=>  C/μ < t_1  <=>  μ > (k·t_d + (1+p)·t_b)/t_1

  记 γ = t_d/t_1（草稿相对成本）、β = t_b/t_1（验证块相对成本）：
      基线条件：  μ > kγ + (1+p)β
      RAISE 条件：μ > kγ + β                ← 消除了 pβ 项

本脚本实测 t_1 / t_d / t_b（同会话、同数据、固定长度、中位数），
输出 γ、β，并给出各剖面「翻正所需的 μ*」——用于判断 A100 上推测解码
究竟能否翻正，以及 RAISE 把门槛降低了多少。

用法：python raise_calib.py --dtype fp16 --k 4 --reps 15 --out results_raise
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

from safespec.spec_engine import load_model, _fwd, _argmax_token, _crop_to
from safespec.data_loader import build_profiles, load_longbench_from_dir, PROFILES

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
TARGET_DEFAULT = os.environ.get("TARGET_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DRAFT_DEFAULT = os.environ.get("DRAFT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "longbench", "data")


def prog(m):
    print("[%s] %s" % (time.strftime("%F %T"), m), flush=True)


def timed_forward(model, input_ids, past, reps, base_len):
    """在固定 KV 长度 base_len 上重复测量前向耗时，取中位数（秒）。

    注意：input_ids 必须是 2 维 [batch, seq]。1 维张量会被模型按
    view(1, -1, head_dim) 切头，导致 q 形状 [1, head_dim, n_heads] 与
    rotary 的 cos [1,1,1,head_dim] 广播错位（RuntimeError at dim 3）。
    """
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
    return statistics.median(ts), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--draft", default=DRAFT_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--profiles", default="all")
    ap.add_argument("--out", default="results_raise")
    ap.add_argument("--wait-free-gb", type=float, default=0)
    args = ap.parse_args()

    dt = DTYPES[args.dtype]
    if args.wait_free_gb > 0:
        while True:
            free = int(os.popen("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
                       .read().strip().split("\n")[0])
            if free >= args.wait_free_gb * 1024:
                break
            prog("等显存 free=%dMiB" % free)
            time.sleep(60)

    prog("加载模型 %s" % args.dtype)
    model, tok = load_model(args.target, dtype=dt)
    draft, _ = load_model(args.draft, dtype=dt)
    tasks = load_longbench_from_dir(DATA_DIR)
    profiles = build_profiles(tasks, tok, max_input=8192)
    want = list(PROFILES) if args.profiles == "all" else args.profiles.split(",")

    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    rows = []
    device = next(model.parameters()).device
    k = args.k

    for prof in [p for p in PROFILES if p in profiles and p in want]:
        ids = profiles[prof]["eval"][0]
        prompt = torch.tensor([ids], device=device, dtype=torch.long)

        # --- target: prefill 后固定长度测单 token 前向 ---
        logits, t_past = _fwd(model, prompt, None)
        nxt = _argmax_token(logits)
        base_t = t_past.get_seq_length()
        t1, _ = timed_forward(model, nxt, t_past, args.reps, base_t)

        # --- target: block(k+1) 前向 ---
        blk = nxt.view(1, 1).repeat(1, k + 1)
        tb, _ = timed_forward(model, blk, t_past, args.reps, base_t)

        # --- draft: 单 token 前向 ---
        dlogits, d_past = _fwd(draft, prompt, None)
        dnxt = _argmax_token(dlogits)
        base_d = d_past.get_seq_length()
        td, _ = timed_forward(draft, dnxt, d_past, args.reps, base_d)

        gamma = td / t1
        beta = tb / t1
        # 翻正所需 μ：基线 μ > kγ + (1+p)β（p 未知，给 p=0 与 p=0.6 两个界）
        mu_base_p0 = k * gamma + beta
        mu_base_p60 = k * gamma + 1.6 * beta
        mu_raise = k * gamma + beta          # RAISE 的 p=0 等价式，与实际 p 无关
        row = dict(profile=prof, ctx_len=len(ids),
                   t1_ms=round(t1 * 1000, 3), td_ms=round(td * 1000, 3),
                   tb_ms=round(tb * 1000, 3),
                   gamma=round(gamma, 4), beta=round(beta, 4),
                   mu_need_base_p0=round(mu_base_p0, 3),
                   mu_need_base_p60=round(mu_base_p60, 3),
                   mu_need_raise=round(mu_raise, 3))
        rows.append(row)
        # 增量落盘：任一剖面出问题也不丢前面已测数据
        with open(os.path.join(HERE, args.out, "raise_calib_%s.json" % args.dtype),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        prog("CALIB %s ctx=%d t1=%.2fms td=%.2fms tb=%.2fms γ=%.3f β=%.3f μ*>%.2f(RAISE)"
             % (prof, len(ids), t1 * 1000, td * 1000, tb * 1000, gamma, beta, mu_raise))

    outdir = os.path.join(HERE, args.out)
    with open(os.path.join(outdir, "raise_calib_%s.json" % args.dtype), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 汇总
    if rows:
        g = statistics.median(r["gamma"] for r in rows)
        b = statistics.median(r["beta"] for r in rows)
        prog("=== 中位：γ=%.3f β=%.3f | RAISE 翻正门槛 μ* = %.2f | 基线(p=0.6) 门槛 μ* = %.2f"
             % (g, b, k * g + b, k * g + 1.6 * b))
    prog("CALIB DONE -> %s/raise_calib_%s.json" % (args.out, args.dtype))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
