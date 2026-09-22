"""RAISE 内核冒烟测试（无需 GPU、无需真实模型、无需网络）。

用确定性 MockLM 模拟 target/draft 的 logits 与 KV cache，验证：
  T1 贪心一致性：raise_decode 输出与 greedy_decode 逐 token 一致（Proposition 1）
  T2 前向账本：每轮 target 前向恰 1 次（M1 消除了冗余 fix 前向）
  T3 EOS 严格性：block 中间出现 EOS 时，终止语义与 AR 一致（M3）
  T4 参数鲁棒：多组 (k, adapt_k, early_stop, margin_tau) 下 T1 均成立
  T5 边界：k_min=k_max=k 固定深度、极小 max_new 不崩

运行：python experiments/raise_smoke_test.py   （或 python -m experiments.raise_smoke_test）
"""
import os
import sys
import hashlib
import random
from types import SimpleNamespace as NS

import torch

# 允许直接以脚本方式运行：把仓库根加入 sys.path，以便 import safespec 包
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from safespec.raise_engine import _argmax_token, _fwd, raise_decode

V = 64
EOS = V - 1


def _det01(seq):
    """确定性哈希 -> [0,1)，避免 PYTHONHASHSEED 不稳定。"""
    s = ",".join(map(str, seq))
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / float(0xFFFFFFFF)


class MockPast:
    """假 KV cache：用 ctx 列表代表已缓存 token，crop 即截断。"""

    def __init__(self):
        self.ctx = []
        self.layers = [self]  # 让 _crop_to 的 for layer in past.layers 生效

    def get_seq_length(self):
        return len(self.ctx)

    def crop(self, n):
        self.ctx = self.ctx[:n]


class MockLM:
    """确定性语言模型：argmax 由真值函数决定，draft 以一定概率猜错。"""

    def __init__(self, is_draft=False, correct_prob=1.0, eos_at=None, seed=0):
        self.is_draft = is_draft
        self.correct_prob = correct_prob
        self.eos_at = eos_at          # 前缀长度达到该值时 target 说 EOS
        self.rng = random.Random(seed)

    def parameters(self):
        return iter([torch.zeros(1)])

    @staticmethod
    def _truth(prefix, eos_at):
        if eos_at is not None and len(prefix) >= eos_at:
            return EOS
        return (prefix[-1] * 3 + len(prefix) * 7) % (V - 1)

    def _logits(self, prefix):
        t = self._truth(prefix, self.eos_at)
        row = [0.0] * V
        row[t] = 10.0
        if self.is_draft:
            if _det01(prefix) >= self.correct_prob:
                alt = (t + 1 + int(_det01(prefix[::-1]) * 5)) % (V - 1)
                row[alt] = 11.0   # draft 猜错 -> argmax 偏离 target
        return row

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        ids = input_ids[0].tolist()
        if past_key_values is None:
            past_key_values = MockPast()
        n_old = len(past_key_values.ctx)
        full = past_key_values.ctx + ids
        past_key_values.ctx = full
        rows = [self._logits(full[: n_old + i + 1]) for i in range(len(ids))]
        return NS(logits=torch.tensor([rows], dtype=torch.float32),
                  past_key_values=past_key_values)


def greedy_ref(target, tok, prompt_ids, max_new):
    """与 spec_engine.greedy_decode 同语义的参考实现。"""
    past = None
    logits, past = _fwd(target, torch.tensor([prompt_ids]), past)
    out = []
    for _ in range(max_new):
        nxt = int(_argmax_token(logits).item())
        out.append(nxt)
        if nxt == tok.eos_token_id:
            break
        logits, past = _fwd(target, torch.tensor([[nxt]]), past)
    return out


def run_case(prompt, max_new, k_init, adapt_k, use_early_stop,
             margin_tau, correct_prob, eos_at, label):
    target = MockLM(is_draft=False, eos_at=eos_at, seed=1)
    draft = MockLM(is_draft=True, correct_prob=correct_prob, eos_at=eos_at, seed=2)
    tok = NS(eos_token_id=EOS)

    ref = greedy_ref(target, tok, prompt, max_new)
    toks, secs, rounds, st = raise_decode(
        target, draft, tok, prompt, max_new=max_new, k_init=k_init,
        adapt_k=adapt_k, use_early_stop=use_early_stop, margin_tau=margin_tau,
        collect_margins=False,
    )

    ok = toks == ref
    # T2: 每轮 target 前向恰 1 次；基线需要 rounds + 失配轮次 次
    fwd_ok = st['target_fwd'] == st['rounds'] and st['target_fwd'] > 0
    base_fwd = st['rounds'] + st['saved_fix_fwd']
    print(f"[{label}] pass={ok} | tokens={len(toks)} rounds={st['rounds']} "
          f"target_fwd={st['target_fwd']} (基线需 {base_fwd}, 省 {st['saved_fix_fwd']}) "
          f"draft_fwd={st['draft_fwd']} avg_k={st['avg_k']:.2f} "
          f"acc/round={st['accept_per_round']:.2f} mismatch={st['rejected']} "
          f"early_stop={st['early_stop']} | fwd_ok={fwd_ok}")
    if not ok:
        n = min(len(toks), len(ref))
        bad = next((i for i in range(n) if toks[i] != ref[i]), n)
        print(f"    !! 首个不一致位置 {bad}: raise={toks[max(0,bad-2):bad+3]} "
              f"ref={ref[max(0,bad-2):bad+3]} (len {len(toks)} vs {len(ref)})")
    return ok and fwd_ok


def main():
    prompt = [3, 17, 42, 8, 55]
    results = []
    print("=" * 78)
    print("T1/T2/T4  多参数贪心一致性 + 前向账本")
    print("=" * 78)
    for cp in (1.0, 0.8, 0.6, 0.4, 0.2):          # draft 质量：从完美到很差
        for k in (2, 4, 6):
            for adapt, es, mt in ((False, False, None), (True, False, None),
                                  (True, True, 0.5), (False, True, 0.5)):
                results.append(run_case(
                    prompt, 40, k, adapt, es, mt, cp, None,
                    f"cp={cp} k={k} adapt={int(adapt)} es={int(es)}"))

    print()
    print("=" * 78)
    print("T3  EOS 严格性（target 在固定长度处输出 EOS）")
    print("=" * 78)
    for eos_at in (12, 20, 31):
        results.append(run_case(prompt, 60, 4, True, False, None, 0.7, eos_at,
                                f"eos_at={eos_at}"))

    print()
    print("=" * 78)
    print("T5  边界：小 max_new / 固定深度")
    print("=" * 78)
    for mn in (1, 2, 3):
        results.append(run_case(prompt, mn, 4, True, False, None, 0.6, None,
                                f"max_new={mn}"))
    results.append(run_case(prompt, 30, 6, True, False, None, 0.5, None,
                            "k_min=k_max=6"))

    print()
    print("=" * 78)
    print(f"结果：{sum(results)}/{len(results)} 通过")
    print("=" * 78)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
