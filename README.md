# safespec-raise

[![smoke-test](https://github.com/Tristan156/safespec-raise/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/Tristan156/safespec-raise/actions/workflows/smoke-test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](CHANGELOG.md)

**长上下文推测解码的收益判据与增量回退调度** —— 参考实现。

<details>
<summary><b>English summary</b>（英文摘要，向下滚动看中文正文）</summary>

Reference implementation of a **falsifiable break-even criterion** for long-context
speculative decoding, together with **RAISE**, a scheduling scheme targeting the dominant
cause of slowdown — the extra target forward pass incurred on mismatched rounds.

Two results matter here:

1. Under the invariant of **exactly one target forward pass per round**, draft depth `k`
   is **not an independent gain lever**. With M1 active, depth drops 4.40 → 2.54 while
   the number of rounds *rises* 9000 → 13648 (+51.6%).
2. The observed slowdown is determined by the **per-round fixed overhead of the execution
   path**, not by the speculative decoding mechanism itself. On an engine with substantially
   lower fixed overhead, the official draft-free n-gram speculative decoding achieves
   **1.163×** speedup.

Of the three RAISE mechanisms, **only M1 yields a positive effect** (M1 +10.69%,
M2a +0.46%, M2b+M3 −11.24%) — reported as measured.

</details>

本仓库是论文《长上下文推测解码的收益判据与增量回退调度》的配套代码与实验脚本。
它给出一个**可证伪的收益判据**，说明在什么条件下推测解码能翻正为加速、在什么条件下
结构性负加速不可能通过调参消除；并给出一个针对负加速主因（失配轮的额外目标前向）的
**调度改进 RAISE**。

> 核心结论一句话：在「每轮恰好一次目标前向」的不变式下，**草稿深度 k 不是独立的收益杠杆**；
> 长上下文负加速的成因是**执行路径的每轮固定开销**，而非推测解码机制本身。

---

## 目录结构

```
safespec-raise/
├── safespec/                    # 核心库（3 个模块，无内部依赖）
│   ├── __init__.py
│   ├── spec_engine.py           # 常规贪心解码 / 块验证推测解码 / 动态 k 基线
│   ├── raise_engine.py          # RAISE 调度内核（M1 + M2 + M3）
│   └── data_loader.py           # LongBench 派生的 10 个请求剖面
├── experiments/                 # 实验脚本（导入上方的 safespec 包）
│   ├── raise_smoke_test.py      # 冒烟测试 / 正确性验证（先跑这个）
│   ├── raise_calib.py           # 标定：测 γ、β、μ、μ*（成本分解）
│   ├── raise_compare.py         # 多方法端到端对照（AR / fixed-k / RAISE）
│   ├── raise_ablation.py        # 机制消融（M1 / M2a / M2b+M3）
│   ├── raise_fair_baseline.py   # 公平基线隔离（四组，排除实现差异）
│   ├── raise_k_sweep.py         # k 消融（k = 2/4/6/8）
│   ├── raise_framework_cmp.py   # 跨框架对照（vs 官方 assisted / lookup）
│   ├── raise_elss.py            # ELSS 早期层自投机对照
│   ├── raise_sweep.py           # 剖面 × 配置扫描
│   ├── raise_analyze.py         # 结果聚合为表（纯离线，只读 jsonl，不需 GPU/torch）
│   └── raise_consistency.py     # 离线一致性核查（同上）
├── docs/                        # 9 份阶段实验结论（已脱敏），入口见 docs/INDEX.md
├── .github/workflows/           # CI：每次推送自动跑冒烟测试
├── data/                        # ← 需自行放置 LongBench（见下），不纳入版本控制
├── requirements.txt
├── pyproject.toml               # 使 safespec 可 pip install -e .
├── CHANGELOG.md                 # 版本变更与实测结论
├── CONTRIBUTING.md              # 贡献规范（含两条不可动摇的不变量）
├── CITATION.cff                 # 引用元数据
├── LICENSE
└── README.md
```

---

## 安装

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 请按你的 CUDA 版本先装 PyTorch，例如 CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

`requirements.txt` 里的 `transformers` 需 **5.x**：本实现依赖 `DynamicCache.layers[i].crop(n)`
（语义为「保留前 n 个 token」）与 `_fwd` 传入的 `past_key_values` 被**原地更新**这两个行为。

### 模型

默认从 Hugging Face 拉取：

| 角色 | 默认模型 | 覆盖方式 |
|---|---|---|
| target | `Qwen/Qwen2.5-7B-Instruct` | `--target` 或环境变量 `TARGET_MODEL` |
| draft | `Qwen/Qwen2.5-0.5B-Instruct` | `--draft` 或环境变量 `DRAFT_MODEL` |

用本地权重时直接传路径即可：

```bash
python experiments/raise_compare.py --target /path/to/Qwen2.5-7B-Instruct \
                                    --draft  /path/to/Qwen2.5-0.5B-Instruct \
                                    --data_dir data/longbench/data
```

### 数据

下载 LongBench 官方数据，放到 `data/longbench/data/`：

```bash
git clone https://github.com/THUDM/LongBench.git /tmp/LongBench
mkdir -p data/longbench && cp -r /tmp/LongBench/data data/longbench/data
```

需要的 10 个 jsonl（`data_loader.py` 负责名称映射，例如论文里的 `samsum-long` 对应
官方文件名 `samsum.jsonl`、`multifieldqa` 对应 `multifieldqa_en.jsonl`）：
`repobench-p, lcc, multi_news, qmsum, samsum, gov_report, narrativeqa, multifieldqa_en, hotpotqa, qasper`。

---

## 快速开始

**1）先跑冒烟测试** —— 无需 GPU、无需模型、无需网络（内部用确定性 MockLM 模拟），
验证 RAISE 与常规贪心解码逐 token 完全一致（无损性，论文 Proposition 1）、
每轮目标前向恒为 1 次（M1）、以及 EOS 终止语义等价（M3）：

```bash
python experiments/raise_smoke_test.py
```

期望末行输出「结果：n/n 通过」。这是**唯一一个不依赖 torch 之外任何东西的验证入口**，
建议在动手跑实验前先确认它通过。

**2）标定成本结构**，得到 γ = t_d/t₁、β = t_b/t₁、μ* = β + kγ：

```bash
python experiments/raise_calib.py --target ... --draft ... --data_dir data/longbench/data \
                                  --out data/calibration.csv
```

**3）端到端对照**：

```bash
python experiments/raise_compare.py --target ... --draft ... --data_dir data/longbench/data
```

结果落在 `experiments/results_*/`（jsonl + 汇总），可用 `raise_analyze.py` 聚合成表。

> `raise_analyze.py` 与 `raise_consistency.py` 是**纯离线后处理**脚本：只读已落盘的 jsonl
> 结果，不加载模型，因此既不需要 GPU 也不需要 torch，可以拿到别的机器上跑。

---

## 方法要点

### 收益判据

记 γ = t_d/t₁（草稿相对成本）、β = t_b/t₁（验证前向相对成本）、μ 为每轮实际接受 token 数，
则每轮成本 μ* = β + kγ，加速比 ≈ μ / μ*。翻正条件是 **μ > μ\***。

参考实现（失配轮需为修正 token 额外付一次目标前向）的判据为式(10)：

```
μ > k·γ + β + p·(γ + 1)        # p 为失配率
```

消除该冗余项后即 RAISE 的判据，式(11)：

```
μ > k·γ + β
```

### RAISE 三机制

| 机制 | 内容 | 作用 |
|---|---|---|
| **M1 增量回退** | 修正 token 不单独前向，而是作为**下一轮 block 的第 0 个输入**，与新一轮草稿一次算完 | 每轮目标前向恒为 1（不含 prefill），消除失配轮的额外 7B 级前向 |
| **M2 置信度门控自适应深度** | 用 draft 每步 top1−top2 logit margin 作该步被接受概率的在线代理；margin 过低时提前终止草稿；并据滑动窗口接受率在线调 k | 不为注定被拒的 token 支付 draft 前向；无需离线契约表，可跨硬件 |
| **M3 严格 EOS 终止** | 对 accepted 做**首个** EOS 截断 | 终止语义与常规自回归贪心一致，消除「长度型不一致」 |

**正确性不变量**：RAISE 只改变**调度**（何时前向、前向多长），不改变**验证语义**
（argmax(target) == draft 才接受）。因此它在贪心解码下与常规解码**逐 token 完全一致**，是无损的。

### 实测结论（论文口径）

- 三机制中**只有 M1 达正效应**：M1 = +10.69%，M2a = +0.46%（噪声级），M2b+M3 = −11.24%。如实报告。
- M1 生效时 `raise_full` 深度 4.40 → 2.54，但**轮数 9000 → 13648（+51.6%）**，验证了
  「每轮一次目标前向」不变式下**深度不是独立收益杠杆**。
- k 消融：k=4 最优（0.7185×），逐剖面 k=4 在 8/10 剖面最优。
- 跨执行路径：官方 transformers 实现（`assisted_generation`、`prompt_lookup_num_tokens=4`）
  落在比本文基线更慢一侧；固定开销显著更低的 vLLM 引擎上，免草稿的官方 n-gram 推测解码
  取得 **1.163×** 正加速。⇒ 负加速由**执行路径的每轮固定开销**决定，非机制本身。

---

## 复现注意事项

- **比值型指标（γ）在共享容器上不可复现**：同配置两次标定可差 50%+。建议只比较**同批次内**
  的相对关系与单调性，不要跨环境引用绝对值。
- 所有实验脚本的计时**不含 prefill**，与基线口径一致。
- 请固定单变量、同预算同 seed 做消融；改代码期间不要与长跑实验并行。
- GPU 建议 ≥24GB（7B target + 0.5B draft，fp16）。CPU 也能跑通冒烟测试，只是很慢。

---

## 引用

本仓库的引用元数据见 [`CITATION.cff`](CITATION.cff)（GitHub 会自动渲染为可复制的 BibTeX）。
配套论文为《长上下文推测解码的收益判据与增量回退调度》，其发表信息将在见刊后补入本文件与
`CITATION.cff`。

---

## 许可

MIT License，见 `LICENSE`。
