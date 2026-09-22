"""ELSS：早期层自投机（Early-Layer Self-Speculation）。

动机（来自内核级剖析 bench_launch.py / bench_fwd.py 的实测）：
    单次前向耗时  t_fwd ≈ n_kernel × τ(启动间隔) + t_gpu
    实测：草稿 Qwen2.5-0.5B 前向 = 1005 个内核，其中 72% 的时间是内核间空隙；
          目标 Qwen2.5-7B   前向 = 1365 个内核。内核数 ≈ 42×层数，即 ∝ 深度。
    ⇒ 推测解码每轮付 (k+1) 次前向，启动开销被放大 (k+1) 倍，
      这正是「长上下文负加速」在共享/虚拟化 GPU 上的真实主因（γ=0.80）。
    ⇒ 降低草稿前向的内核数 = 降低 γ。最直接的办法：草稿只用目标模型的前 L 层
      （早期退出），内核数随 L/28 线性下降，且**无需额外草稿模型**。

本脚本扫描 L ∈ {2,4,6,8,12,16,24,28}：
  * 记录每档 L 的草稿前向内核数、t_d、γ、以及
  * 同会话运行 AR 锚点与 RAISE-fixed4（草稿=前 L 层），实测 μ 与加速比，
  * 给出「加速比-L」曲线与最优 L（γ 随 L 线性上升、μ 随 L 饱和 ⇒ 存在最优 L）。

纪律：不改动任何既有代码与结果；输出写入 results_elss/。

用法：
  python raise_elss.py --lens 2,4,6,8,12,16,24,28 --profiles qmsum,gov_report,lcc \
      --nreq 12 --k 4 --ctx 2048 --dtype fp16
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
OUT_DIR = os.path.join(HERE, "results_elss")
T_MAX_NEW = int(os.environ.get("ELSS_MAX_NEW", "128"))


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


# ------------------------------------------------------------ 早期层草稿构建
def build_early_layers(target_path, L, dtype):
    """用目标模型的前 L 层 + LM head 构造草稿模型（不加载其余层，省显存）。"""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(target_path)
    total = getattr(cfg, "num_hidden_layers", None)
    cfg.num_hidden_layers = min(L, total or L)
    m = AutoModelForCausalLM.from_pretrained(
        target_path, config=cfg, dtype=dtype, device_map="cuda",
        low_cpu_mem_usage=True)
    m.eval()
    return m, total


# ------------------------------------------------------------ 计时与标定
def timed_forward(model, input_ids, past, reps, base_len):
    if input_ids.dim() == 1:
        input_ids = input_ids.view(1, -1)
    ts = []
    for _ in range(reps):
        _crop_to(past, base_len)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _logits, past = _fwd(model, input_ids, past)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    _crop_to(past, base_len)
    return statistics.median(ts)


@torch.no_grad()
def calibrate(model, draft, tok, prompt_ids, k, reps, want_kernel=False):
    dev = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], device=dev, dtype=torch.long)
    t_logits, t_past = _fwd(model, ids, None)
    d_logits, d_past = _fwd(draft, ids, None)
    base_t, base_d = t_past.get_seq_length(), d_past.get_seq_length()
    nxt = _argmax_token(t_logits)
    t1 = timed_forward(model, nxt, t_past, reps, base_t)
    td = timed_forward(draft, nxt, d_past, reps, base_d)
    blk = nxt.view(1, 1).repeat(1, k + 1)
    tb = timed_forward(model, blk, t_past, reps, base_t)
    out = dict(ctx=int(base_t), t1_ms=t1 * 1e3, td_ms=td * 1e3, tb_ms=tb * 1e3,
               gamma=td / t1 if t1 else None, beta=tb / t1 if t1 else None)
    if out["gamma"]:
        out["mu_star"] = k * out["gamma"] + out["beta"]
        out["ub"] = (k + 1) / out["mu_star"]
    if want_kernel:
        try:
            from torch.profiler import profile, ProfilerActivity
            b1 = nxt.view(1, 1)
            for _ in range(3):
                _crop_to(d_past, base_d)
                _fwd(draft, b1, d_past)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    _crop_to(d_past, base_d)
                    _fwd(draft, b1, d_past)
                torch.cuda.synchronize()
            _crop_to(d_past, base_d)
            ev = prof.key_averages()
            out["draft_kernels"] = round(sum(e.count for e in ev
                                             if getattr(e, "device_time", None)) / 5, 1)
        except Exception as e:
            out["draft_kernels_err"] = repr(e)
    return out


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_DEFAULT)
    ap.add_argument("--dtype", default="fp16", choices=list(DTYPES))
    ap.add_argument("--lens", default="2,4,6,8,12,16,24,28")
    ap.add_argument("--profiles", default="qmsum,gov_report,lcc")
    ap.add_argument("--nreq", type=int, default=12)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--methods", default="ar,raise_fixed4")
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    want = [p.strip() for p in args.profiles.split(",") if p.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    dt = DTYPES[args.dtype]

    prog("加载目标模型")
    model, tok = load_model(args.target, dtype=dt)
    n_total = model.config.num_hidden_layers
    tasks = load_longbench_from_dir(DATA_DIR)
    profs = build_profiles(tasks, tok, max_input=args.ctx)
    use = [p for p in PROFILES if p in profs and p in want]
    prog("目标层数=%d 剖面=%s" % (n_total, use))
    calib_rows = []

    for L in lens:
        try:
            draft, _tot = build_early_layers(args.target, L, dt)
        except Exception as e:
            prog("L=%d 构建早期草稿失败: %r" % (L, e))
            continue
        nL = draft.config.num_hidden_layers
        nparam = sum(p.numel() for p in draft.parameters())
        prog("=== L=%d (实际 %d 层, %.3fB 参数) ===" % (L, nL, nparam / 1e9))

        first_prof = use[0]
        req0 = profs[first_prof]["eval"][:args.nreq]
        try:
            cb = calibrate(model, draft, tok, req0[0], args.k, args.reps, want_kernel=True)
            cb.update(L=L, layers=nL, params=nparam, profile=first_prof)
            calib_rows.append(cb)
            prog("  标定 L=%d: ctx=%d t1=%.2fms td=%.2fms tb=%.2fms γ=%.4f β=%.4f "
                 "μ*=%.3f 上限=%.3f× 草稿内核数=%.0f"
                 % (L, cb["ctx"], cb["t1_ms"], cb["td_ms"], cb["tb_ms"], cb["gamma"],
                    cb["beta"], cb.get("mu_star", float("nan")), cb.get("ub", float("nan")),
                    cb.get("draft_kernels", -1)))
        except Exception as e:
            prog("  标定失败 L=%d: %r" % (L, e))

        for prof in use:
            reqs = profs[prof]["eval"][:args.nreq]
            for method in methods:
                name = "L%d_%s_%s_%s" % (L, method, args.dtype, prof)
                if cell_done(name, len(reqs)):
                    prog("  SKIP %s" % name)
                    continue
                recs = []
                for i, ids in enumerate(reqs):
                    try:
                        if method == "ar":
                            toks, secs = greedy_decode(model, tok, ids, T_MAX_NEW)
                            st = {"target_fwd": len(toks)}
                        else:
                            toks, secs, _rounds, st = raise_decode(
                                model, draft, tok, ids, max_new=T_MAX_NEW, k_init=args.k,
                                adapt_k=False, use_early_stop=False, collect_margins=False)
                    except Exception as e:
                        prog("  ERR %s req%d: %r" % (name, i, e))
                        continue
                    recs.append(dict(req=i, method=method, L=L, layers=nL, profile=prof,
                                     dtype=args.dtype, prompt_len=len(ids),
                                     n_tokens=len(toks), seconds=round(secs, 5),
                                     tpot_ms=round(secs / max(1, len(toks)) * 1e3, 4),
                                     target_fwd=st.get("target_fwd"),
                                     rounds=st.get("rounds"),
                                     accept_per_round=st.get("accept_per_round"),
                                     saved_fix_fwd=st.get("saved_fix_fwd"),
                                     tokens=toks))
                write_cell(name, recs)
                if recs:
                    tp = sum(r["seconds"] for r in recs) / sum(r["n_tokens"] for r in recs) * 1e3
                    mu = [r["accept_per_round"] for r in recs if r.get("accept_per_round")]
                    prog("  DONE %s n=%d tpot=%.3fms μ=%.3f"
                         % (name, len(recs), tp, statistics.mean(mu) if mu else float("nan")))
        del draft
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(OUT_DIR, "calib_elss.json"), "w", encoding="utf-8") as f:
        json.dump(calib_rows, f, ensure_ascii=False, indent=1)
    prog("ALL DONE -> results_elss/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
