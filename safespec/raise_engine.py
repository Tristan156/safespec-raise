"""RAISE: Rejection-Aware Incremental Speculative Execution.

SafeSpec-Cert v2 的调度内核。相对原 greedy block-verification 推测解码
（spec_engine.speculative_decode / dynamic_k_decode），本文件修正三处结构性浪费：

  M1 增量回退 (Incremental Rollback)
     基线：失配 -> crop KV -> 对修正 token 再做 1 次 target 前向 (fix 前向)。
     本实现：修正 token 不单独前向，而是作为【下一轮 block 的第 0 个输入】
     与新一轮草稿一起前向，一次算完。每轮 target 前向次数恒为 1（不含 prefill），
     消除了失配轮次额外的 7B 级前向——这是 A100 负加速的主要来源之一。

  M2 置信度门控自适应深度 (Confidence-Gated Adaptive Depth)
     用 draft 每步 top1-top2 logit margin 作为该步被 target 接受概率的在线代理，
     margin 过低时提前终止草稿，避免为注定被拒的 token 支付 draft 前向；
     并据滑动窗口接受率在线调整 k（无需离线契约表，天然可跨硬件）。

  M3 严格 EOS 终止 (EOS-Consistent Termination)
     基线只检查 block 最后一个 token 是否为 EOS，block 中间出现 EOS 时会继续
     生成，产生"长度型不一致"（H4 实测长度不一致的主导来源）。本实现对 accepted
     做首个 EOS 截断，使终止语义与常规自回归贪心解码严格一致。

正确性不变量（贪心语义，Proposition 1）：
    在 greedy 验证下，RAISE 生成的 token 序列与 greedy_decode 逐 token 完全一致。
    RAISE 只改变【调度】（何时前向、前向多长），不改变【验证语义】
    （argmax(target) == draft 才接受）。故它是无损的（lossless）。

用法（与 spec_engine 兼容）：
    model, tok = load_model(target_path)
    draft, _   = load_model(draft_path)
    toks, secs, rounds, stats = raise_decode(model, draft, tok, prompt_ids, k_init=4)
"""
import time
import torch


# ---------------------------------------------------------------- 基础算子
# 与 spec_engine.py 语义逐字对齐（transformers 5.x：传入的 past 会被原地更新）
@torch.no_grad()
def _fwd(model, input_ids, past):
    out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
    return out.logits, out.past_key_values


def _crop_to(past, length):
    """截断 KV cache 到 length（保留前 length 个 token，原地）。"""
    if past is None:
        return past
    for layer in past.layers:
        layer.crop(length)
    return past


def _argmax_token(logits):
    return torch.argmax(logits[:, -1, :], dim=-1)


def _seq_len(past):
    return past.get_seq_length() if past is not None else 0


# ---------------------------------------------------------------- 主循环
@torch.no_grad()
def raise_decode(model, draft, tok, prompt_ids, max_new=128, k_init=4,
                 k_min=2, k_max=6, win=16, margin_tau=None,
                 adapt_k=True, use_early_stop=False, eos_strict=True,
                 collect_margins=True):
    """RAISE 推测解码。

    Args:
        k_init       : 初始草稿深度
        k_min/k_max  : 自适应深度边界（adapt_k=True 时生效）
        win          : 接受率滑动窗口长度
        margin_tau   : 置信度门控阈值；为 None 时不启用早停
        adapt_k      : 是否在线调整 k（False = 固定 k_init，用于单变量消融）
        use_early_stop: 是否启用 margin 早停（M2）
        eos_strict   : 是否做首个 EOS 严格截断（M3）
    Returns:
        tokens  : 生成 token 列表
        seconds : 解码耗时（不含 prefill，与基线口径一致）
        rounds  : 每轮 (本轮接受 token 数, 本轮 k)，与 spec_engine 同格式
        stats   : 前向次数与调度统计（用于开销归因）
    """
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    # prefill（不计入 TPOT，与基线一致）
    t_logits, t_past = _fwd(model, prompt, None)
    d_logits, d_past = _fwd(draft, prompt, None)

    generated = []
    rounds = []
    margins_log = []
    st = dict(target_fwd=0, draft_fwd=0, accepted=0, rejected=0,
              early_stop=0, rounds=0, saved_fix_fwd=0, k_sum=0)
    window = []
    k = int(k_init)
    fix = None          # 上一轮失配产生的 target 修正 token，待并入下一轮 block
    t0 = time.time()

    while len(generated) < max_new:
        base_len = _seq_len(t_past)      # KV 已覆盖长度（注意 fix 尚未写入）

        # ---------------- M2: draft 生成（含 margin 门控） ----------------
        draft_tokens = []
        if fix is not None:
            # draft 侧消费上一轮的修正 token（0.5B 侧，廉价；target 侧不重复支付）
            fx = torch.tensor([[fix]], device=device, dtype=torch.long)
            dlog, d_past = _fwd(draft, fx, d_past)
            st['draft_fwd'] += 1
        else:
            dlog = d_logits

        for _ in range(k):
            last = dlog[0, -1, :].float()
            if collect_margins or (use_early_stop and margin_tau is not None):
                top2 = torch.topk(last, 2).values
                margin = float(top2[0] - top2[1])
            else:
                margin = None
            if margin is not None:
                margins_log.append(margin)
            # 至少保留 1 个草稿 token；margin 过低时停止追加注定被拒的 token
            if (use_early_stop and margin_tau is not None
                    and margin is not None and margin < margin_tau
                    and len(draft_tokens) >= 1):
                st['early_stop'] += 1
                break
            dnext = torch.argmax(last, dim=-1, keepdim=True)   # [1,1]
            draft_tokens.append(int(dnext.item()))
            dlog, d_past = _fwd(draft, dnext.unsqueeze(0) if dnext.dim() == 1 else dnext, d_past)
            st['draft_fwd'] += 1

        k_eff = len(draft_tokens)

        # ---------------- M1: block 前向（修正 token 并入） ----------------
        if fix is not None:
            block = [fix] + draft_tokens
            off = 1
        else:
            block = list(draft_tokens)
            off = 0
        cand = torch.tensor([block], device=device, dtype=torch.long)
        t_new_logits, t_past = _fwd(model, cand, t_past)
        st['target_fwd'] += 1

        # ---------------- 贪心验证 ----------------
        if fix is None:
            first_pred = int(_argmax_token(t_logits).item())
        else:
            first_pred = int(torch.argmax(t_new_logits[0, 0, :]).item())

        accepted = []
        new_fix = None
        mismatch = False
        if first_pred == draft_tokens[0]:
            accepted.append(draft_tokens[0])
            for i in range(k_eff - 1):
                pred = int(torch.argmax(t_new_logits[0, off + i, :]).item())
                if pred == draft_tokens[i + 1]:
                    accepted.append(draft_tokens[i + 1])
                else:
                    accepted.append(pred)
                    new_fix = pred
                    mismatch = True
                    break
        else:
            accepted.append(first_pred)
            new_fix = pred = first_pred
            mismatch = True

        # 统计
        st['rounds'] += 1
        st['k_sum'] += k_eff
        if mismatch:
            st['rejected'] += 1
            st['saved_fix_fwd'] += 1        # 基线会在此多付 1 次 target 前向
        st['accepted'] += len(accepted)
        rounds.append((len(accepted), k_eff))
        window.append(len(accepted))
        if len(window) > win:
            window = window[-win:]

        # ---------------- KV 对齐 ----------------
        # 有效前缀 = base_len + off(修正token) + 已接受数
        # 【关键】失配时 accepted 的最后一个元素是修正 token，它【尚未写入 KV】，
        # 将由下一轮 block 的第 0 个输入承担；因此裁剪时须把它排除（-1）。
        keep = base_len + off + len(accepted) - (1 if mismatch else 0)
        if keep < _seq_len(t_past):
            _crop_to(t_past, keep)
        if keep < _seq_len(d_past):
            _crop_to(d_past, keep)

        # 全接受时，block 末行 logits 即下一位置的预测，无需重新前向
        if not mismatch:
            t_logits = t_new_logits[:, -1:, :]
            d_logits = dlog
        fix = new_fix

        # ---------------- M3: 严格 EOS 终止 ----------------
        if eos_strict and tok.eos_token_id in accepted:
            cut = accepted.index(tok.eos_token_id) + 1
            generated.extend(accepted[:cut])
            break
        generated.extend(accepted)
        if not accepted:
            break

        # ---------------- M2: 在线调整 k ----------------
        if adapt_k and len(window) >= win:
            acc_rate = sum(window) / (len(window) * max(1, k))
            if acc_rate > 0.75 and k < k_max:
                k = min(k_max, k + 2)
            elif acc_rate < 0.45 and k > k_min:
                k = max(k_min, k - 2)

    t1 = time.time()
    if len(generated) > max_new:
        generated = generated[:max_new]
    st['avg_k'] = st['k_sum'] / max(1, st['rounds'])
    st['accept_per_round'] = st['accepted'] / max(1, st['rounds'])
    st['margins'] = margins_log
    return generated, (t1 - t0), rounds, st
