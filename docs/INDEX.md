# 文档索引

`docs/` 收录的是各阶段实验的**结论记录**，不是教程。阅读顺序建议按「先结论 → 后细节」。

> 说明：本目录下的文档已做**环境标识脱敏**。文中的 `<HOST>`、`<PROJ_ROOT>`、`<MODELS_DIR>`
> 分别代表实验服务器地址、工程根目录、模型权重目录；「实验服务器」「实验容器」为脱敏后的
> 环境称谓。这些占位符不影响任何结论，读者只需把它们替换成自己的环境即可。

---

## 一、先读这两份（核心结论）

| 文档 | 内容 | 对应论文部分 |
|---|---|---|
| [why-long-context-cannot-break-even.md](why-long-context-cannot-break-even.md) | **收益判据分析**：为什么长上下文推测解码在数学上无法翻正。由实测的 γ / β / μ 推出翻正条件，并给出「负加速不可通过调参消除」的论证 | 收益判据（式 10 / 式 11） |
| [STAGE_RESULTS_v3_ablation.md](STAGE_RESULTS_v3_ablation.md) | **机制消融**：M1 / M2a / M2b+M3 三机制各自贡献的隔离测量。结论是只有 M1 达正效应 | RAISE 三机制 |

## 二、方法横向对比

| 文档 | 内容 |
|---|---|
| [method_compare.md](method_compare.md) | 四方法横向对比（fp16）：AR / SD-k / SDc-k / SD-dyn，同数据集、同硬件、同会话口径 |
| [method_compare_bf16.md](method_compare_bf16.md) | 同上，bf16 精度下重跑，用于核对精度是否改变结论 |

> 口径：`speedup = AR_TPOT / 方法_TPOT`（>1 表示加速）；`risk_n` 为长度型不一致请求数（一致性风险）。
> AR = 常规自回归（分母）；SD-k = 固定 k 推测解码（Leviathan 式）；SDc-k = 轮末钳位固定 k；
> SD-dyn = 接受率驱动动态 k。

## 三、单变量扫描

| 文档 | 内容 |
|---|---|
| [STAGE_RESULTS_v5_ksweep.md](STAGE_RESULTS_v5_ksweep.md) | 推测深度 k 的单变量扫描（k = 2/4/6/8）。结论：k=4 最优，逐剖面看 k=4 在 8/10 剖面最优 |

## 四、跨执行路径 / 跨框架

| 文档 | 内容 |
|---|---|
| [STAGE_RESULTS_v4_framework.md](STAGE_RESULTS_v4_framework.md) | 与主流推理库**官方公开实现**的同条件对照（`assisted_generation`、`prompt_lookup` 等），用于分离「机制」与「执行路径」的影响 |
| [STAGE_RESULTS_vllm.md](STAGE_RESULTS_vllm.md) | vLLM 0.29.0 的离线安装与冒烟测试记录。关键证据：固定开销显著更低的引擎上，官方 n-gram 推测解码取得 **1.163×** 正加速 |
| [vllm-offline-install.md](vllm-offline-install.md) | 上一条的**完整离线安装操作记录**（独立 venv，不改动主环境），供离线环境复现 |

## 五、一致性归因

| 文档 | 内容 |
|---|---|
| [STAGE_RESULTS_diag2.md](STAGE_RESULTS_diag2.md) | 内容型分歧的**精度归因**（复算批次）：区分「数值精度导致的分歧」与「真实语义分歧」 |

---

## 已剔除的文档（不在本仓库）

为保持仓库只含**纯技术结论**，以下类别的文档未纳入公开仓库：投稿策略与目标刊评估、
审稿往来与回应信、改稿排期、内部架构改造排期。
这些内容与代码实现无关，且涉及未公开的投稿状态。

---

## 文件名变更记录

公开时对两份中文文件名做了 ASCII 化，避免跨平台解压/检出时出现编码问题。内容未改动：

| 原文件名 | 现文件名 |
|---|---|
| `vLLM离线安装记录.md` | `vllm-offline-install.md` |
| `收益判据分析_为什么长上下文无法翻正.md` | `why-long-context-cannot-break-even.md` |
