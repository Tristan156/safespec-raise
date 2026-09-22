# STAGE RESULTS — vLLM 0.29.0 离线安装与冒烟测试

**执行时间**：2026-09-21 15:38–18:10
**执行位置**：实验服务器 实验容器（`ssh <HOST>`）
**本机副本**：`safespec_compare/vllm/`（脚本 + 全部日志 + 环境清单）
**总体状态**：**部分成功** —— 安装 ✅ / 冒烟 ✅ / 官方推测解码机制 ✅ / **用户指定的 0.5B→7B 草稿对 ❌ 被 vLLM 词表校验硬拒**
**硬约束遵守**：`safespec` 基线环境 **0 改动**（102 包逐行 diff 为空，本机独立复核确认）；未安装 torch/transformers/numpy；未 kill 任何他人进程；未发邮件。

---

## 0　一句话结论

**vLLM 0.29.0 已在容器内离线装成并跑通推理（16 token 正常生成），官方推测解码的执行链路也已验证可用；但用户指定的 `Qwen2.5-0.5B → Qwen2.5-7B` 草稿/目标对无法启动——不是环境问题，而是 vLLM 在配置校验阶段就拒绝了词表不一致的组合（152064 vs 151936），该拒绝发生在任何显存分配之前，改硬件/加显存都无效。**

替代方案已验证可用（见 §6）：**vLLM 官方 n-gram（prompt-lookup）推测解码无需草稿模型，已实测 `NGRAM_TEST_PASS`**，可绕开词表校验直接构成"官方 vLLM 推测解码"同条件基线。

---

## 1　交付物与路径

| 项 | 路径 | 说明 |
|---|---|---|
| **独立 venv** | `<VLLM_ENV>` | 228 包，`--system-site-packages` 建在 `safespec` 之上 |
| **wheel 仓** | `<WHEEL_DIR>` | 182 wheel / 1.3 G，纯离线安装源 |
| **运行入口** | `<WHEEL_DIR>/vllm_python.sh` | **必须用这个包装器跑**，内含 4 处必要环境变量（见 §8） |
| 冒烟脚本 | `vllm_wheels/smoke_test.py` | 16 token 生成验证 |
| 草稿模型测试 | `vllm_wheels/spec_test.py` | 参数化 `TARGET`/`DRAFT`，异常落盘 |
| n-gram 测试 | `vllm_wheels/spec_ngram_test.py` | 无草稿模型的官方推测解码 |
| **官方基线跑分器** | `vllm_wheels/official_baseline.py` | **下一步做对比实验直接用它**，内含 GPU 显存闸门 |
| 判据驱动脚本 | `vllm_wheels/run_spec_pair.sh` | 一次跑完 A/B 两组对照 |

本机副本：`safespec_compare/vllm/`（含 `smoke_final.log`、`spec_05b_to_7b.log`、`spec_05b_to_05b.log`、`spec_ngram.log`、`official_baseline_dryrun.json`、`_base_pkgs_before.txt` / `_base_now.txt` / `_venv_now.txt`）。

---

## 2　安装结果：**成功**

匹配性在安装前已逐项核实（vLLM 0.29.0 要求 `torch==2.13.0`，与容器**精确匹配**，这是"可行"判断的唯一依据）：

| 项 | vLLM 0.29.0 要求 | 容器实际 | |
|---|---|---|---|
| torch | `==2.13.0` | 2.13.0+cu130 | ✅ |
| transformers | `>=5.10.4` | 5.15.0 | ✅ |
| Python | `>=3.10,<3.15` | 3.10.20 | ✅ |
| glibc | `manylinux_2_28`（≥2.28） | Ubuntu 20.04 / **2.31** | ✅ |
| 驱动 | — | 590.48.01 | ✅ |

最终版本表（venv 内实测）：

| 包 | 版本 | 备注 |
|---|---|---|
| vllm | **0.29.0** | |
| torch | 2.13.0+cu130 | 复用基线，未重装 |
| transformers | 5.15.0 | 复用基线，未重装 |
| tokenizers | 0.22.2 | **回退到基线版本**（见 §8 坑 7） |
| triton | 3.7.1 | **回退到基线版本**（见 §8 坑 7） |
| numpy | 2.2.6 | 复用基线，未重装 |
| llguidance | 1.7.6 | 镜像缺失，走官方 PyPI 补下 |
| xgrammar | 0.2.7 | 环境标记跳过，单独补下 |
| uvloop | 0.22.1 | **vLLM 元数据漏声明**，见 §8 坑 6 |
| flashinfer-python | 0.6.18 | JIT 编译成功 |

**包数**：venv 共 **228** 个包，其中相对基线**新增 144** 个。

**残留告警（唯一一条，已知可接受）**：

```
tokenspeed-mla 0.1.8 requires tokenspeed-triton, which is not installed.
```

`tokenspeed-triton` 是 Blackwell（SM100/103）专用的 Triton 私有 fork，PyPI 上不存在（`post<日期>` 后缀指向私有索引）。本机是 A100/SM80，不需要它；且真装上去会**遮蔽 torch 自带的 triton**，风险大于收益，故有意不装。

---

## 3　冒烟测试：**通过**

命令：
```bash
cd <WHEEL_DIR> && ./vllm_python.sh smoke_test.py
```

实测输出（`smoke_final.log`，逐字）：

```
>>> vllm 0.29.0 | torch 2.13.0+cu130 | cuda=True
>>> vllm 导入耗时 5.94 s
>>> 模型加载耗时 10.66 s
>>> 生成耗时 0.253 s
PROMPT : 'The capital of France is'
OUTPUT : ' Paris. It is the largest city in Europe and the third largest city in the'
TOKENS : 16 个 -> [12095, 13, 1084, 374, 279, 7772, 3283, 304, 4505, 323, 279, 4843, 7772, 3283, 304, 279]
>>> 合计: 加载 10.66s + 生成 0.253s
>>> SMOKE_TEST_PASS
```

引擎侧实测口径（同次运行）：

| 指标 | 值 |
|---|---|
| 注意力后端 | FLASH_ATTN |
| 权重占用 | 0.93 GiB |
| KV cache | 2.43 GiB / **212,448 tokens** |
| 最大并发 | 103.73×（2048 token/请求） |
| 引擎 init（profile+KV+warmup） | 3.06 s |
| 采样器 | FlashInfer top-p/top-k |
| KV layout | LBNHC |

**结论：离线安装的 vLLM 可以正常加载本地模型并生成，推理链路完整可用。**

---

## 4　推测解码启动结果

三组测试，全部用同一脚本族、同一 `gpu_memory_utilization=0.05`、`enforce_eager=True`：

| # | 组合 / 方法 | 结果 | 启动耗时 | 生成耗时 | 证据文件 |
|---|---|---|---|---|---|
| **A** | draft=**0.5B** → target=**7B**（用户指定） | ❌ **失败**（exit=1） | — | — | `spec_05b_to_7b.log` |
| **B** | draft=0.5B → target=0.5B（同词表对照） | ✅ **通过** | 11.75 s | 1.060 s | `spec_05b_to_05b.log` |
| **C** | method=**ngram**（无草稿模型） | ✅ **通过** | 10.48 s | 1.543 s | `spec_ngram.log` |

### A　请求的组合：启动失败，报错原文

```
INFO 09-21 18:07:10 [speculative.py:1586] Overriding draft model max model len from 32768 to 2048
>>> SPEC_START_FAILED
>>> 异常类型: ValidationError
>>> 异常信息: 1 validation error for SpeculativeConfig
  Value error, Target and draft model should have the same vocabulary size.
  Target model vocab_size=152064. Draft model vocab_size=151936.
  Using models with different tokenizers can cause out-of-bounds errors
  during speculative decoding. [type=value_error, ...]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
```

调用栈定位（说明**失败发生在配置校验阶段，早于任何显存/权重分配**）：

```
spec_test.py:20  ->  LLM(...)
vllm/entrypoints/llm.py:343        ->  LLMEngine.from_engine_args
vllm/v1/engine/llm_engine.py:171   ->  create_engine_config
vllm/engine/arg_utils.py:2387      ->  create_speculative_config
vllm/engine/arg_utils.py:1952      ->  SpeculativeConfig(**self.speculative_config)
pydantic_core.ValidationError
```

### B　同词表对照：通过

```
>>> 推测解码引擎启动成功, 耗时 11.75 s
OUTPUT : ' Paris. It is the largest city in Europe and the third largest city in the'
TOKENS : 16 个
>>> 生成耗时 1.060 s
>>> SPEC_TEST_PASS
```

同时观察到 EAGLE 系列 Triton 内核被实际编译调用（`eagle_prepare_next_token_padded_kernel`、`copy_and_expand_eagle_inputs_kernel`、`eagle_step_slot_mapping_metadata_kernel`、`eagle_prepare_inputs_padded_kernel`），说明**推测解码的执行链路是完整跑通的，不只是配置通过**。

> 注：B 组生成文本与冒烟测试**逐字一致**（greedy, temp=0），是一个有效的正确性旁证。

### C　n-gram（无草稿模型）：通过

```
>>> method=ngram | num_speculative_tokens=5 | gmu=0.050
>>> ngram 推测解码引擎启动成功, 耗时 10.48 s
>>> 生成耗时 1.543 s
>>> NGRAM_TEST_PASS
```

---

## 5　失败根因

**根因：vLLM 的 `SpeculativeConfig` 强制要求草稿与目标模型词表尺寸严格相等，而 Qwen2.5 系列内部就不一致。**

| 模型 | `vocab_size` | 模型体积 |
|---|---|---|
| Qwen2.5-0.5B-Instruct | **151936** | 954 M |
| Qwen2.5-7B-Instruct | **152064** | 29 G |
| Qwen3-32B | 151936 | 62 G |

**值得记录的事实**：两个模型的 `tokenizer.json` **逐字节完全相同**（md5 均为 `3f99a313c3754208a7f3acfccadc53cd`）。即**词元 ID 空间是同一套**，7B 只是把嵌入矩阵多补了 128 行（152064 = 151936 + 128，补齐到 1188×128 的整数倍以便张量核对齐）。因此 vLLM 报的 "different tokenizers" 在本例中**字面上不成立**，它用的是"词表尺寸相等"这一更保守的代理判据。

但**不建议**绕过该校验（打补丁 / monkeypatch）：vLLM 源码注释明确指出词表不一致会导致 KV 索引越界；即便本例中可论证安全，一篇要投 Q1 的论文若建立在"patch 掉安全校验"的官方基线上，审稿人会直接质疑对照的有效性。**正确做法是换基线路线，不是拆校验。**

**次要阻塞（叠加）**：GPU 为共享卡，实测空闲 8715 MiB（总量 81920 MiB）。7B fp16 仅权重就约 15 GB，**即使词表问题解决，当前显存下也起不来**——需等空闲 ≥ 20 GiB。

---

## 6　对"与官方 vLLM 推测解码同条件对比"的影响与可行路线

| 路线 | 可行性 | 代价 | 建议 |
|---|---|---|---|
| **① 官方 n-gram / prompt-lookup 推测解码** | ✅ **已实测通过** | 需换草稿模型或改用 ngram | **首选**。无需草稿模型、无词表校验、显存开销最小；长上下文与代码类高重复负载下正是官方推测解码的真实形态，与本文 LongBench 场景高度契合 |
| ② 换同词表草稿/目标对 | ⚠️ 受限 | 容器的 `Qwen3-32B`（151936）与 0.5B 同词表，但 62 GB 起不来 | 需另行下载小体积同词表对（如 1.5B/3B 级别） |
| ③ 打补丁绕过词表校验 | ❌ 不推荐 | 破坏官方实现，审稿风险 | 不作为论文对照 |
| ④ 放弃官方对照，仅与自研基线比 | ⚠️ 说服力弱 | 缺"主流引擎"参照 | 仅作退路 |

**关键判断**：路线 ① **今天就可以在 7B 上跑**（只等显存），并且它天然满足"官方 vLLM 官方推测解码"这一对照定义。

---

## 7　环境安全审计：`safespec` 基线 **0 改动**

| 核查项 | 结果 |
|---|---|
| 基线包数（安装前 / 现在） | 102 / **102** |
| `diff _base_pkgs_before.txt _base_now.txt` | **空（0 差异行）** |
| venv 包数 | 228 |
| venv 相对基线新增 | 144 |
| 是否重装 torch / transformers / numpy | **否**（均复用基线） |
| 是否触碰他人 GPU 进程 | **否**（全程未 kill 任何 PID） |

本机独立复核（不依赖容器内判断）：对回收的 `_base_pkgs_before.txt` 与 `_base_now.txt` 重新 diff，结果同样为 **IDENTICAL**。

> venv 的 `bin/python` 是指向 `safespec` conda python 的符号链接（`--system-site-packages` 机制），这正是必须用 `vllm_python.sh` 包装器补 `LD_LIBRARY_PATH` 的原因（见 §8 坑 3）。

---

## 8　踩坑与修复清单（9 项，**下次直接绕开**）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | `pip download` 漏下 `xgrammar`/`llguidance`/`tokenspeed-mla` | Windows 上 pip 按**本机**平台评估环境标记，Linux 专属依赖被静默跳过 | 按包名**单独补下**（不带标记即不评估），加 `--no-deps` |
| 2 | 镜像上 `llguidance` 最高 1.6.1（需 ≥1.7.0）、`tokenspeed-mla==0.1.8` 查不到 | 阿里云镜像**落后**于官方 PyPI | 改走官方 PyPI，并加 `--platform manylinux_2_31_x86_64` |
| 3 | `from vllm import LLM` **段错误 (139)** | venv 的 python 是符号链接，丢失 conda 的 `$ORIGIN/../lib` RPATH → 加载系统 `libstdc++` 时 `CXXABI_1.3.15` 缺失（`libicui18n.so.78` 需要） | **`LD_LIBRARY_PATH=<HOME>/dev/miniforge3/envs/safespec/lib`**（核心修复） |
| 4 | `ModuleNotFoundError: uvloop` | `uvloop` **未在 vLLM 0.29.0 元数据中声明**，却被 `vllm/v1/utils.py` 无条件 import（上游打包缺陷） | 补装 `uvloop-0.22.1`（另预防性补 `msgpack`、`orjson`） |
| 5 | `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | 容器无系统 CUDA | `CUDA_HOME=.../nvidia/cu13`（nvcc 13.4.92），并把 `$CUDA_HOME/bin` 加入 PATH |
| 6 | `FileNotFoundError: 'ninja'` | 包装器直接 exec python，`venv/bin` 不在 PATH | PATH 前置 `<VLLM_ENV>/bin` |
| 7 | FlashInfer JIT 编译：`curand.h: No such file`；链接期 `cannot find -lcudart` | venv 的 `nvidia/cu13` 头文件(84)与库目录不全（缺 74 个头、缺 `lib64`、缺 `.so` 开发符号链接） | 从基线软链 74 个缺失头文件；`ln -sfn libcudart.so.13 libcudart.so` 等；`ln -sfn lib lib64`（**仅改 venv**） |
| 8 | `tokenizers>=0.22.0,<=0.23.0 required, found 0.23.2`；`triton==3.7.1` 冲突 | venv 里装的版本破坏了基线 pin（transformers 5.15.0 与 torch 2.13.0 的要求） | venv 内 `pip uninstall -y tokenizers triton` 回退到基线 0.22.2 / 3.7.1；`fsspec` 同理回退 |
| 9 | `Free memory ... less than desired` | 共享 GPU，vLLM 默认 `gpu_memory_utilization=0.9` 会预分配整卡 90% | 显式压到 `0.05`；`official_baseline.py` 内置显存闸门，不足直接退出 |

---

## 9　下一步建议（按优先级）

**P0｜等显存空闲 ≥ 20 GiB，跑通 7B 上的官方 n-gram 基线**（今天即可，只差显存）：

```bash
cd <WHEEL_DIR>
TARGET=<MODELS_DIR>/Qwen2.5-7B-Instruct \
METHOD=ngram NUM_SPEC_TOKENS=5 GPU_MEM_UTIL=0.30 MAX_MODEL_LEN=8192 \
MIN_FREE_GIB=20 \
  ./vllm_python.sh official_baseline.py
```
> `official_baseline.py` 已实测通过（0.5B 干跑：GPU 闸门 → ngram 引擎 10.22 s → 2 个剖面 → JSON 落盘 → `OFFICIAL_BASELINE_PASS`）。显存不足时会打印 `GPU_GUARD_ABORT` 并退出，**不会挤占他人任务**。

**P1｜把对照组口径钉死**（论文可比性的前提）：固定 `gpu_memory_utilization` / `enforce_eager` / `max_model_len` / 剖面集合，并明确 vLLM 的引擎级 warmup 与 KV 预分配如何从计时中扣除，使口径与自研实现一致。

**P2｜补一个无草稿的官方 AR 锚点**：同脚本 `METHOD=none`，构成「vLLM 自回归 vs vLLM 推测解码」的引擎内增益，与「自研 AR vs 自研 RAISE」形成对称对照。

**P3｜若要保留 draft-model 路线**：需另下**同词表**的小体积草稿对（Qwen3-32B 虽同词表但 62 GB 不可用），并在选择前先用 `config.json` 的 `vocab_size` 做匹配筛查——**这是选草稿模型的第一道硬门槛**。

---

## 10　原始证据索引

| 文件 | 内容 |
|---|---|
| `vllm/smoke_final.log` | 冒烟测试完整日志（`SMOKE_TEST_PASS`） |
| `vllm/spec_05b_to_7b.log` | 用户指定组合的完整报错原文与调用栈 |
| `vllm/spec_05b_to_05b.log` | 同词表对照通过日志（`SPEC_TEST_PASS`） |
| `vllm/spec_ngram.log` | n-gram 官方推测解码通过日志（`NGRAM_TEST_PASS`） |
| `vllm/official_baseline_dryrun.json` | 基线跑分器干跑结果（含 2 剖面耗时/吞吐） |
| `vllm/run_spec_pair.out` | A/B 两组退出码（A=1，B=0） |
| `vllm/_base_pkgs_before.txt` / `_base_now.txt` | 基线环境前后清单（diff 为空） |
| `vllm/pipcheck3.log` | 唯一残留告警原文 |
| `vllm/vllm_python.sh` | 运行入口（4 处必需环境变量） |
| `vllm/official_baseline.py` | 官方基线跑分器（含显存闸门） |

---

*本阶段无邮件发送（按要求结果并入最终总报告）。*
