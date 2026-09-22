# STAGE RESULTS v4 — 框架层同条件对照（与主流推理库官方公开实现）

> 实验完成时间：2026-09-21 22:03（远端日志 `=== END ===` 收尾）
> 数据源：实验容器 `<PROJ_ROOT>/experiments/results_framework/framework_summary.json`
> 本机副本：`safespec_compare/framework/framework_summary.json`
> 脚本：`raise_framework_cmp.py`（新增文件，未改动任何既有代码与结果）
> 环境：A100-80G 共享容器，fp16，Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct，transformers 5.15.0 / torch 2.13.0+cu130

---

## 0　一句话结论

**官方实现 `hf_assisted` 的同条件加速比为 0.2992×，落在 `fixed4`（0.6512×）一侧，而不是 `fair_fixed4`（0.7187×）一侧。**
⇒ 本文原基线"失配轮多付一次目标前向"的冗余**并非本文实现特有**；官方实现在同条件下同样负加速，且绝对水平还**低于**本文基线约 2.2 倍。
⇒ **论文口径已由"仅本文参考实现路径"上移一档**：负加速成因与判据形式的结论按**跨实现成立**给出；增益幅值（+10.3%）与一致性改善幅值仍限于与本文同代码路径的比较。

---

## 1　实验协议

| 项 | 设定 |
|---|---|
| 对照物 | ① `hf_assisted` = transformers 官方 `model.generate(assistant_model=draft, do_sample=False)`；② `hf_lookup` = 官方 `prompt_lookup_num_tokens=4`（免草稿模型的 n-gram 检索式推测解码） |
| 五组 | `ar`（同会话自回归锚点）/ `fixed4`（原基线）/ `fair_fixed4`（仅 M1）/ `hf_assisted` / `hf_lookup` |
| 规模 | 10 剖面 × 8 请求 × 2 遍 = 160 请求/方法，逐请求轮转（方法顺序随请求序号轮换） |
| 解码 | fp16，k=4，生成上限 128（`min_new_tokens` 同设，避免提前 EOS 干扰） |
| 计时口径 | 自研方法：内部计时（天然不含 prefill）；HF 方法：`(T_generate − T_prefill)/n_tokens`，`T_prefill` 由对同一 prompt 的一次原始前向（`use_cache=True`）单测 |
| 一致率口径 | 以同会话自回归输出为锚点；因轮转导致部分请求的 AR 锚点在同遍内不可用，一致率在"存在 AR 锚点"的可比子集上统计（各方法可比样本数不同，见表） |

**vLLM / SGLang / TensorRT-LLM：端到端同条件数据均未取得**，且原因各不相同（2026-09-21 18:10 交叉核对，**修正了先前"未安装/不兼容"的笼统说法**）：

| 框架 | 真实状态 | 证据 |
|---|---|---|
| **vLLM 0.29.0** | **已离线装成**（独立 venv `<VLLM_ENV>`，182 wheel 离线源），官方推理与 **n-gram 推测解码链路已跑通**（`NGRAM_TEST_PASS`）；但本文所用的 **0.5B→7B 草稿-目标对被词表尺寸校验硬拒**（152064 vs 151936，拒绝先于任何显存分配，加显存无效）；**7B 官方 n-gram 实测尚未取数**（需共享卡空闲 ≥20 GiB，实测仅 12.3 GiB） | `vllm_wheels/install.log`、`smoke_final.log`、`spec_ngram.log`、`spec_05b_to_7b.log`、`official_baseline_dryrun.json`（**仅 0.5B 干跑，非可比基准**）、`nvidia-smi` |
| **SGLang / TensorRT-LLM** | 未预装，发行版与容器内 torch 2.13.0+cu130 运行时绑定 → 确不可行 | — |

**本文只做机制层与文献报告值对照，未给出也未合成任何端到端数据。** 论文 §3.11 与 §4 已按上表的准确原因表述（不再写"未预装/不兼容"这种笼统且与事实不符的理由）。

---

## 2　五组对照表（同批、同协议）

| 方法 | 实现来源 | 加权 TPOT/ms | 加速比 vs AR | 目标前向 | 修正前向 | 完全一致率 |
|---|---|---|---|---|---|---|
| ar | 本文锚点 | 38.065 | 1.0000× | 20480 | 0 | — |
| fixed4 | 本文参考实现（原基线） | 58.457 | **0.6512×** | 7530 | 4200 | 33.57% |
| fair_fixed4 | 本文 RAISE（仅 M1） | 52.964 | **0.7187×** | 7542 | 0 | 90.83% |
| hf_assisted | transformers 官方 assisted generation | 127.245 | **0.2992×** | — ¹ | — ¹ | 16.36% |
| hf_lookup | transformers 官方 prompt-lookup（k=4） | 83.804 | **0.4542×** | — ¹ | — ¹ | 16.00% |

¹ HF 两组的每轮目标前向次数**未在脚本内插桩**，故此列留空；本文不对此作任何推断。

**交叉校验**：`ar` 目标前向 20480 = 160 请求 × 128 token，与生成上限口径自洽；`fixed4` 修正前向 4200 次（失配修正），`fair_fixed4` 为 0，与 M1 机制定义一致。

**官方实现落点量化**：`fixed4 / hf_assisted` = 0.6512 / 0.2992 = **2.18 倍**（即本文基线比官方实现快约 2.2 倍）。
**辅助对照**：`hf_lookup`（免草稿模型路线）0.4542× 亦低于 `fixed4`，但高于 `hf_assisted`。

---

## 3　逐剖面相对加速比

分母 = 该剖面同会话自回归均值。

| 剖面 | ar | fixed4 | fair_fixed4 | hf_assisted | hf_lookup |
|---|---|---|---|---|---|
| gov_report | 1.0000 | 0.5770 | 0.6454 | 0.4299 | 0.5111 |
| hotpotqa-long | 1.0000 | 0.6435 | 0.7130 | 0.3426 | 0.5054 |
| lcc | 1.0000 | 0.8088 | 0.8757 | 0.2929 | 0.3125 |
| multi_news | 1.0000 | 0.5420 | 0.6244 | 0.2601 | 0.3188 |
| multifieldqa | 1.0000 | 0.7540 | 0.8212 | 0.4645 | 0.6322 |
| narrativeqa | 1.0000 | 0.5648 | 0.6388 | 0.3909 | 0.4731 |
| qasper | 1.0000 | 0.5408 | 0.6055 | 0.3749 | **0.5590** |
| qmsum | 1.0000 | 0.7996 | 0.8359 | 0.1284 | 0.7139 |
| repobench-p | 1.0000 | 0.8330 | 0.9006 | 0.3253 | 0.3629 |
| samsum-long | 1.0000 | 0.5969 | 0.6473 | 0.3816 | 0.4788 |

**逐剖面统计（全部由实测推导）**

| 判据 | 结果 |
|---|---|
| `fair_fixed4` 在五组中最快的剖面数 | **10/10** |
| `hf_assisted` 慢于 ar 且慢于 fixed4 的剖面数 | **10/10** |
| `hf_lookup` 慢于 fixed4 的剖面数 | **9/10**（仅 `qasper` 略快：0.5590× vs 0.5408×） |
| `fixed4 > hf_assisted` 的剖面数 | **10/10** |
| 方向一致性 | 逐剖面方向一致，总体判读不由个别剖面拉动 |

注意 `qmsum` / `lcc` / `multi_news` 三个剖面上 `hf_assisted` 跌破 0.13~0.30×，是官方实现在长 prompt + 短生成场景下固定开销占比最高的形态。

---

## 4　口径处理（**本次的核心改动**）

| 项 | 改动前 | 改动后 | 依据 |
|---|---|---|---|
| 负加速成因（§3.6） | "仅本文参考实现路径" | **跨实现成立** | 官方实现在同条件下负加速（0.2992×） |
| 判据形式（§1.7） | "仅本文参考实现路径" | **跨实现成立** | 同上；未翻正 ⇒ μ<μ* 在官方实现上亦成立 |
| 增益幅值（+10.3%） | 已限本文路径 | **维持**限本文代码路径 | 幅值依赖实现路径，不跨实现搬运 |
| 一致性改善幅值 | 已限本文路径 | **维持** | 含生成预算/提交口径处理差异 |
| 反向情形（某实现已按式(11)消除该项） | "部分推理库的既有做法"（推测语气） | 改为"**本文实测的两套官方实现均未呈现这一有利特征**，但该情形作为部署前提仍应逐案核实" | 实测证伪了"官方默认已规避"的预设 |
| 不越界声明 | — | 新增：HF 内部计数未插桩 ⇒ 判读基于端到端可观测口径，**不对官方实现内部计数作推断** | 计量口径诚实性 |

**正文落点（均已重建入库）**

| 位置 | 改法 |
|---|---|
| §3.11（新增小节） | 五组对照（表20）+ 逐剖面表（表21）+ 不可实测框架的原因说明 + 三条边界说明 |
| §3.7 | 口径段改写：幅值限本文路径 / 负加速结论跨实现；保留式(11)逐案核实 |
| §3.10 | "唯一同条件比较是 4 种调度策略"→ 改为指向 §3.7 与 §3.11 |
| 摘要（中/英） | 增补同条件对照结论 |
| 贡献 g) | 增补同条件对照结论 |
| 结语 | 增补"该判断已由两种官方公开实现独立佐证" |
| §4（部署比较段 / 适用范围段） | 增补同条件对照落点 + vLLM/SGLang/TensorRT-LLM 不可实测的原因与"未合成数据"声明 |

---

## 5　表号校验

| 检查项 | 结果 |
|---|---|
| 文档表格对象数 | 21 |
| 中文表题数 / 英文表题数 | 21 / 21（一一对应） |
| 中文表号序列 | `[1..21]` |
| 是否唯一且严格递增 | **True** |
| 中英表号一致 | **True** |
| 正文引用覆盖 | 1~21 全部被引用，**无悬空引用** |
| 新增表号 | 表20（五组同条件对照，`take_no()` 动态分配）、表21（逐剖面相对加速比，`take_no()` 动态分配） |
| 工具 | `_tools/check_v14_tables.py` + `_tools/v14_qa.py`（式号 1~12 无缺号、图号 1~4 一致、无占位符） |

---

## 7　复现命令

```bash
# 远端（实验容器）
docker exec <CONTAINER> bash -lc 'cd <PROJ_ROOT>/experiments && tail -60 logs/fw.log'   # 末尾应有 === END ===
# 回收
scp <HOST>:<PROJ_ROOT>/experiments/results_framework/framework_summary.json \
  "C:/Users/lenovo/WorkBuddy/2026-08-18-08-31-44/safespec_compare/framework/framework_summary.json"
# 重建与校验
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe build_v14.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/check_v14_tables.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/v14_qa.py
```

---

## 8　官方置信度门控的跨实现对照（2026-09-22 补全定稿）

### 8.1 目的与口径

本节把“与主流推理库的对比”从实现级推进到**机制级跨实现**：用 transformers 官方的**置信度门控**（`assistant_confidence_threshold=0.4`，配合 `num_assistant_tokens=4` 与 `num_assistant_tokens_schedule="heuristic"`）与本文 **M2a**（以 top1/top2 logit margin 为判据的自适应深度）做**同层对照**——二者作用层完全对应，都是“按置信度决定推测深度”的在线调度机制，差别仅在于实现方。

**实验条件**：实验容器，脚本 `raise_framework_cmp.py`，输出 `results_conf/`，日志 `logs/conf.log`；fp16、`--limit 8 --repeats 2`（每剖面 8 请求 × 2 遍 = 16 条/方法），10 个剖面，三方法（`ar` / `hf_assisted` 官方固定深度 / `hf_conf` 官方置信度门控）共 **480 条记录**。

**比值定义**：`hf_conf 归一化耗时 / hf_assisted 归一化耗时`（各以 `seconds/n_tokens` 归一化后逐请求配对），**>1 表示官方门控更快**。

### 8.2 配对检验结果（全部 10 个剖面，覆盖完整）

| 指标 | 数值 |
|---|---|
| 配对数 n | **160** |
| 覆盖剖面 | **10 / 10**（gov_report、hotpotqa-long、lcc、multi_news、multifieldqa、narrativeqa、qasper、qmsum、repobench-p、samsum-long） |
| 比值中位 | **0.9944** |
| 95% 自助置信区间 | **[0.9896, 1.0117]**（区间跨 1） |
| 门控更快（比值>1） | 71 / 160 |
| 符号检验 p 值 | 0.179 |
| 是否显著 | **否** |

**逐剖面方向**（比值 >1 表示门控更快）：

| 剖面 | 比值 | 方向 |
|---|---|---|
| gov_report | 1.0076 | 门控更快 |
| hotpotqa-long | 1.0520 | 门控更快 |
| lcc | 0.9922 | 门控更慢 |
| multi_news | 0.9985 | 门控更慢 |
| multifieldqa | 0.9868 | 门控更慢 |
| narrativeqa | 0.9721 | 门控更慢 |
| qasper | 1.0191 | 门控更快 |
| qmsum | 1.0493 | 门控更快 |
| repobench-p | 0.9971 | 门控更慢 |
| samsum-long | 1.0238 | 门控更快 |

**方向统计**：为正 **5/10**（gov_report、hotpotqa-long、qasper、qmsum、samsum-long）；为负 **5/10**（lcc、multi_news、multifieldqa、narrativeqa、repobench-p）。逐剖面方向**双向分散**，不存在一致方向 ⇒ 总体判读不由个别剖面拉动。

### 8.3 与本文 M2a 的对照结论

| 对照项 | 本文 M2a | 官方置信度门控 | 是否一致 |
|---|---|---|---|
| 作用层 | margin 判据自适应深度 | `assistant_confidence_threshold` 自适应深度 | **完全对应** |
| 效应量 | +0.56%（比值中位 1.0056） | **0.9944**（比值中位） | 同量级、同符号侧（均≈1） |
| 统计可辨性 | 噪声级 | **不显著**（CI 跨 1，p=0.179） | **一致** |
| 逐剖面方向 | 混向 | **混向（5/10 为正、5/10 为负）** | **一致** |

**结论（定案）**：两种实现（本文与官方）在**“置信度门控不构成吞吐杠杆”**这一点上结论一致。该判断因此**并非本文参考实现的产物**，而是与式(11) 的成本结构相符的一般现象——在 μ 已低于阈值 μ\* 的配置下，缩短推测深度同时压低每轮有效推进量，其代价与收益同量级，无法把结论推向有利侧。

> **口径纪律**：本轮结论与 §3.11（框架层同条件对照）属**同一档口径**——把“机制判断”上移到跨实现成立，而**增益幅值**仍限于本文同代码路径。本条**不**用于支持任何跨实现的幅值声明。

### 8.4 待办与边界

| # | 项 | 说明 |
|---|---|---|
| 1 | 官方门控组每轮目标前向/修正前向**未插桩** | 与 §3.11 的 HF 两组同一边界：判读基于端到端可观测口径，**不对官方实现的内部计数作推断**。若需升级为机制级计数证据，属新增实验（非本轮范围） |
| 2 | 单剖面耗时极不均衡 | qmsum 剖面因 HF assisted 约 541 ms/token，单剖面约 45 分钟；全 10 剖面 08:45 启动、11:21 收尾（约 2 h 36 min），日志以 `=== END ===` 正常收尾 |
| 3 | 本轮样本量 | `--limit 8 --repeats 2` → 16 条/剖面/方法，与 §3.11 的 160 请求/方法同量级，配对检验效力充足 |

### 8.5 复现命令

```bash
# 远端（实验容器）
ssh <HOST> 'cd <PROJ_ROOT>/experiments && tail -40 logs/conf.log'   # 末尾应有 === END ===
# 回收
scp 'root@<HOST>:<PROJ_ROOT>/experiments/results_conf/*.jsonl' \
  'C:/Users/lenovo/WorkBuddy/2026-08-18-08-31-44/safespec_compare/conf/'
# 刷新对比 + 重建 + 校验
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/conf_compare.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe build_v14.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/v14_qa.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/v14_check.py
```

> **注**：`_tools/conf_compare.py` 的自助法已核实为**每轮抽 n 个**（非此前的每次抽 10000 个）。独立复核：正确写法 CI 宽度 ≈0.0220（两个随机种子下 0.0220 / 0.0221，稳定）；退化的 buggy 写法 CI 宽度仅 0.0017。**本条 CI 有效，非退化解。**

---

## 8　官方置信度门控的跨实现对照（2026-09-22 补全定稿）

### 8.1 目的与口径

本节把“与主流推理库的对比”从实现级推进到**机制级跨实现**：用 transformers 官方的**置信度门控**（`assistant_confidence_threshold=0.4`，配合 `num_assistant_tokens=4` 与 `num_assistant_tokens_schedule="heuristic"`）与本文 **M2a**（以 top1/top2 logit margin 为判据的自适应深度）做**同层对照**——二者作用层完全对应，都是“按置信度决定推测深度”的在线调度机制，差别仅在于实现方。

**实验条件**：实验容器，脚本 `raise_framework_cmp.py`，输出 `results_conf/`，日志 `logs/conf.log`；fp16、`--limit 8 --repeats 2`（每剖面 8 请求 × 2 遍 = 16 条/方法），10 个剖面，三方法（`ar` / `hf_assisted` 官方固定深度 / `hf_conf` 官方置信度门控）共 **480 条记录**。

**比值定义**：`hf_conf 归一化耗时 / hf_assisted 归一化耗时`（各以 `seconds/n_tokens` 归一化后逐请求配对），**>1 表示官方门控更快**。

### 8.2 配对检验结果（全部 10 个剖面，覆盖完整）

| 指标 | 数值 |
|---|---|
| 配对数 n | **160** |
| 覆盖剖面 | **10 / 10**（gov_report、hotpotqa-long、lcc、multi_news、multifieldqa、narrativeqa、qasper、qmsum、repobench-p、samsum-long） |
| 比值中位 | **0.9944** |
| 95% 自助置信区间 | **[0.9896, 1.0117]**（区间跨 1） |
| 门控更快（比值>1） | 71 / 160 |
| 符号检验 p 值 | 0.179 |
| 是否显著 | **否** |

**逐剖面方向**（比值 >1 表示门控更快）：

| 剖面 | 比值 | 方向 |
|---|---|---|
| gov_report | 1.0076 | 门控更快 |
| hotpotqa-long | 1.0520 | 门控更快 |
| lcc | 0.9922 | 门控更慢 |
| multi_news | 0.9985 | 门控更慢 |
| multifieldqa | 0.9868 | 门控更慢 |
| narrativeqa | 0.9721 | 门控更慢 |
| qasper | 1.0191 | 门控更快 |
| qmsum | 1.0493 | 门控更快 |
| repobench-p | 0.9971 | 门控更慢 |
| samsum-long | 1.0238 | 门控更快 |

**方向统计**：为正 **5/10**（gov_report、hotpotqa-long、qasper、qmsum、samsum-long）；为负 **5/10**（lcc、multi_news、multifieldqa、narrativeqa、repobench-p）。逐剖面方向**双向分散**，不存在一致方向 ⇒ 总体判读不由个别剖面拉动。

### 8.3 与本文 M2a 的对照结论

| 对照项 | 本文 M2a | 官方置信度门控 | 是否一致 |
|---|---|---|---|
| 作用层 | margin 判据自适应深度 | `assistant_confidence_threshold` 自适应深度 | **完全对应** |
| 效应量 | +0.56%（比值中位 1.0056） | **0.9944**（比值中位） | 同量级、同符号侧（均≈1） |
| 统计可辨性 | 噪声级 | **不显著**（CI 跨 1，p=0.179） | **一致** |
| 逐剖面方向 | 混向 | **混向（5/10 为正、5/10 为负）** | **一致** |

**结论（定案）**：两种实现（本文与官方）在**“置信度门控不构成吞吐杠杆”**这一点上结论一致。该判断因此**并非本文参考实现的产物**，而是与式(11) 的成本结构相符的一般现象——在 μ 已低于阈值 μ\* 的配置下，缩短推测深度同时压低每轮有效推进量，其代价与收益同量级，无法把结论推向有利侧。

> **口径纪律**：本轮结论与 §3.11（框架层同条件对照）属**同一档口径**——把“机制判断”上移到跨实现成立，而**增益幅值**仍限于本文同代码路径。本条**不**用于支持任何跨实现的幅值声明。

### 8.4 待办与边界

| # | 项 | 说明 |
|---|---|---|
| 1 | 官方门控组每轮目标前向/修正前向**未插桩** | 与 §3.11 的 HF 两组同一边界：判读基于端到端可观测口径，**不对官方实现的内部计数作推断**。若需升级为机制级计数证据，属新增实验（非本轮范围） |
| 2 | 单剖面耗时极不均衡 | qmsum 剖面因 HF assisted 约 541 ms/token，单剖面约 45 分钟；全 10 剖面 08:45 启动、11:21 收尾（约 2 h 36 min），日志以 `=== END ===` 正常收尾 |
| 3 | 本轮样本量 | `--limit 8 --repeats 2` → 16 条/剖面/方法，与 §3.11 的 160 请求/方法同量级，配对检验效力充足 |

### 8.5 复现命令

```bash
# 远端（实验容器）
ssh <HOST> 'cd <PROJ_ROOT>/experiments && tail -40 logs/conf.log'   # 末尾应有 === END ===
# 回收
scp 'root@<HOST>:<PROJ_ROOT>/experiments/results_conf/*.jsonl' \
  'C:/Users/lenovo/WorkBuddy/2026-08-18-08-31-44/safespec_compare/conf/'
# 刷新对比 + 重建 + 校验
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/conf_compare.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe build_v14.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/v14_qa.py
C:/Users/lenovo/.workbuddy/binaries/python/envs/default/Scripts/python.exe _tools/v14_check.py
```

> **注**：`_tools/conf_compare.py` 的自助法已核实为**每轮抽 n 个**（非此前的每次抽 10000 个）。独立复核：正确写法 CI 宽度 ≈0.0220（两个随机种子下 0.0220 / 0.0221，稳定）；退化的 buggy 写法 CI 宽度仅 0.0017。**本条 CI 有效，非退化解。**

---

## 9. 硬伤清零与口径修正（2026-09-22 14:30）

### 9.1 官方置信度门控对照：6 剖面 → 全 10 剖面（已完成）
自动化 `6518c069` 于 13:30 完成补全。10 个剖面 jsonl 全部落盘（每剖面 16 条 = 8 请求 × 2 遍），
配对总数 **160**。

| 指标 | 值 |
|---|---|
| 配对数 n | **160**（10 剖面） |
| 比值中位 | **0.9944** |
| 95% CI | **[0.9896, 1.0117]**（跨 1 ⇒ 不显著） |
| 门控更快 | **71/160** |
| 双侧符号检验 p | **0.179** |
| 逐剖面为正 | **5/10** |

逐剖面：正 — gov_report 1.0076、hotpotqa-long 1.0520、qasper 1.0191、qmsum 1.0493、samsum-long 1.0238；
负 — lcc 0.9922、multi_news 0.9985、multifieldqa 0.9868、narrativeqa 0.9721、repobench-p 0.9971。

**判读**：官方门控跨全部 10 个剖面**同样不构成吞吐杠杆**，与本文 M2a（ratio_median 1.0056）同层同结论
⇒ 该判断非本文参考实现的产物。已数据驱动并入 §3.7。

### 9.2 三处口径矛盾修正（真硬伤，已修）
1. **§3.11 导语**原写「vLLM、SGLang 与 TensorRT-LLM 的端到端同条件实测**均未取得数据**」，
   与本节末尾 vLLM 官方 n-gram 已取得 1.163× 正加速的结论**自相矛盾**。改为：
   未取得的仅 SGLang 与 TensorRT-LLM；vLLM 的免草稿 n-gram 路线已取得端到端数据，
   仅「0.5B→7B 草稿-目标对」因词表尺寸校验被拒（152064 vs 151936，拒绝先于显存分配）。
2. **§4 适用范围段**原文有句内矛盾（「vLLM 未取得数据：vLLM 已离线装成并取得端到端同条件数据，
   但 n-gram 推测解码链路，但……」），含重复「但」与语义断裂。已重写为通顺且与 §3.11 一致的表述。
3. **§4 部署比较段**同步补入 vLLM 1.163×（10/10 剖面为正）并按「执行路径每轮固定开销」口径收口；
   「对上述三者」改为「对 SGLang 与 TensorRT-LLM 二者」。

### 9.3 段落拆分
§3.11 导语由单段 870 字拆为两段（可实测范围 / 对照物选取），降低单段长度。

### 9.4 复核结果
- 全局过时表述扫描（「对上述三者」「均未取得数据」「vLLM 未预装」「扩样到 47」「2-2.5」「ICML」）
  **零命中**。
- 关键数字各就其位：1.163×4、0.2992×5、0.4542×4、0.6512×5、0.7187×2、0.9944×1、10.3%×6、5797×6。
- 重建：**305 段 / 25 表**；`v14_qa.py` 与 `v14_check.py` **双 exit 0**。
- 摘要与结语均已含 vLLM 正加速结论，口径与正文一致。

### 9.5 明确「不是硬伤」的两项（避免误报）
1. **§3.9 的「24 点与 25 点」正确**。经复核 `safespec_compare/raise/diag_precision_flip.json`
   的 24 条记录 = 24 个唯一 `(profile,req,d)`，首批本就已去重；与 `results_ablation/divergence_points.json`
   的「47 条原始记录 / 22 个唯一」是**两批不同数据**，不可混用。
2. **贡献列举段（para23 605 字、para31 635 字）保持整段**，属学术论文惯用的「其一/其二」列举式写法，
   拆分会破坏编号连贯性。

### 9.6 剩余非硬伤项（排版偏好，未处理）
正文仍有 9 段超 450 字（最长 726 字，为 §3.7 口径澄清段）。属可读性优化而非科学性问题，
若目标刊排版要求严格可再拆。
