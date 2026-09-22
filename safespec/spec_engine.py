"""SafeSpec-Cert 核心引擎：推测解码 + 常规解码 + TPOT 测量 + 一致性检查。

标准 greedy speculative decoding（block verification）。
关键实现事实（transformers 5.x）：
  - _fwd 传入的 past_key_values 会被【原地更新】，返回的与传入的是同一对象；
  - DynamicCache 的 layer.crop(n) 语义是【保留前 n 个 token】（截断到长度 n）。
验证语义：
  - 验证第 1 个候选 c1 用「预测 prompt 之后」的 target logits；
  - 验证第 j 个候选 c{j}（j>=2）用 target 对 [c1..c{k}] 一次前向结果的 logits[0, j-2]。
"""
import time
import torch


def load_model(path, device="cuda", dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    return model, tok


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
    return torch.argmax(logits[:, -1, :], dim=-1)  # shape [batch]


def greedy_decode(model, tok, prompt_ids, max_new=128):
    """常规解码 (k=0)。返回 (token序列, 生成耗时秒)。"""
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    logits, past = _fwd(model, input_ids, None)  # prefill，不计入 TPOT
    generated = []
    t0 = time.time()
    nxt = _argmax_token(logits)
    for _ in range(max_new):
        tid = nxt.item()
        generated.append(tid)
        if tid == tok.eos_token_id:
            break
        logits, past = _fwd(model, nxt.unsqueeze(1), past)
        nxt = _argmax_token(logits)
    t1 = time.time()
    return generated, (t1 - t0)


def speculative_decode(model, draft, tok, prompt_ids, k, max_new=128):
    """推测解码 (greedy block verification)。返回 (token序列, 耗时秒, 每轮(接受数,候选数))。"""
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    t_logits, t_past = _fwd(model, prompt, None)   # t_logits 预测 prompt 后
    d_logits, d_past = _fwd(draft, prompt, None)   # d_logits 预测 prompt 后

    generated = []
    rounds = []
    t0 = time.time()

    while len(generated) < max_new:
        base_len = t_past.get_seq_length()

        # 1) 草稿自回归生成 k 个候选（原地更新 d_past）
        draft_tokens = []
        dlog = d_logits
        for _ in range(k):
            dnext = _argmax_token(dlog)
            draft_tokens.append(dnext.item())
            dlog, d_past = _fwd(draft, dnext.unsqueeze(1), d_past)

        # 2) 目标 block 验证前向（原地更新 t_past）
        cand = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        t_new_logits, t_past = _fwd(model, cand, t_past)

        # 3) 贪心验证
        accepted = []
        stop = False
        first_pred = _argmax_token(t_logits).item()
        if first_pred == draft_tokens[0]:
            accepted.append(draft_tokens[0])
            for i in range(k - 1):
                pred = torch.argmax(t_new_logits[0, i, :]).item()
                if pred == draft_tokens[i + 1]:
                    accepted.append(draft_tokens[i + 1])
                else:
                    accepted.append(pred)
                    stop = True
                    break
        else:
            accepted.append(first_pred)
            stop = True

        rounds.append((len(accepted), k))
        generated.extend(accepted)
        if generated[-1] == tok.eos_token_id:
            break

        # 4) 更新 KV 到 base_len + len(accepted)
        if stop:
            m = len(accepted) - 1
            _crop_to(t_past, base_len + m)   # 回退到 prompt + c1..cm
            _crop_to(d_past, base_len + m)
            fix = torch.tensor([[accepted[-1]]], device=device, dtype=torch.long)
            t_logits, t_past = _fwd(model, fix, t_past)
            d_logits, d_past = _fwd(draft, fix, d_past)
        else:
            t_logits = t_new_logits[:, -1:, :]  # 预测 prompt+accepted 之后
            d_logits = dlog

    t1 = time.time()
    return generated, (t1 - t0), rounds


def dynamic_k_decode(model, draft, tok, prompt_ids, max_new=128, init_k=4,
                     win_size=32, up_thresh=0.6, down_thresh=0.3):
    """档1 基线：接受率驱动动态 k，按最近窗口平均接受率在 {2,4,6} 间调整。"""
    device = next(model.parameters()).device
    prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)

    t_logits, t_past = _fwd(model, prompt, None)
    d_logits, d_past = _fwd(draft, prompt, None)

    generated = []
    window = []
    k = init_k
    t0 = time.time()

    while len(generated) < max_new:
        base_len = t_past.get_seq_length()

        draft_tokens = []
        dlog = d_logits
        for _ in range(k):
            dnext = _argmax_token(dlog)
            draft_tokens.append(dnext.item())
            dlog, d_past = _fwd(draft, dnext.unsqueeze(1), d_past)

        cand = torch.tensor([draft_tokens], device=device, dtype=torch.long)
        t_new_logits, t_past = _fwd(model, cand, t_past)

        accepted = []
        stop = False
        first_pred = _argmax_token(t_logits).item()
        if first_pred == draft_tokens[0]:
            accepted.append(draft_tokens[0])
            for i in range(k - 1):
                pred = torch.argmax(t_new_logits[0, i, :]).item()
                if pred == draft_tokens[i + 1]:
                    accepted.append(draft_tokens[i + 1])
                else:
                    accepted.append(pred)
                    stop = True
                    break
        else:
            accepted.append(first_pred)
            stop = True

        if stop:
            window.extend([1] * (len(accepted) - 1) + [0])
        else:
            window.extend([1] * len(accepted))
        if len(window) > win_size:
            window = window[-win_size:]

        generated.extend(accepted)
        if generated[-1] == tok.eos_token_id:
            break

        if stop:
            m = len(accepted) - 1
            _crop_to(t_past, base_len + m)
            _crop_to(d_past, base_len + m)
            fix = torch.tensor([[accepted[-1]]], device=device, dtype=torch.long)
            t_logits, t_past = _fwd(model, fix, t_past)
            d_logits, d_past = _fwd(draft, fix, d_past)
        else:
            t_logits = t_new_logits[:, -1:, :]
            d_logits = dlog

        if len(window) >= win_size:
            rate = sum(window) / len(window)
            if rate > up_thresh:
                k = 6 if k == 4 else (4 if k == 2 else k)
            elif rate < down_thresh:
                k = 2 if k == 4 else (4 if k == 6 else k)

    t1 = time.time()
    return generated, (t1 - t0)


def seq_consistent(a, b):
    """逐 token 一致性：长度 + 每个位置完全一致。"""
    return a == b
