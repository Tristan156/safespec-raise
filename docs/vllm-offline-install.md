# vLLM 0.29.0 离线安装记录（离线实验容器）

> 目的：在**完全离线**的共享服务器上装出可与主流程并存的 vLLM，用于"与主流推理引擎官方推测解码同条件对比"。
> 结论：**安装成功**（`import vllm` 通过，vllm 0.29.0，2.8s）。冒烟测试与推测解码启动由后续自动化补记。
> 原则：**绝不改动现有可用的 `safespec` 环境**；全部装进独立 venv。

---

## 1　为什么之前判断"不可行"，后来又"可行"

| 时间 | 判断 | 依据 |
|---|---|---|
| 15:36 前 | 不可行 | 仅基于"未安装 + 容器 torch 2.13.0 版本过新"的**推测** |
| 15:40 后 | **可行** | 实际查 PyPI：vLLM 0.29.0 要求 **`torch==2.13.0`**，与容器**精确匹配** |

> 教训：**可行性结论必须以实际查询为准，不能凭印象下结论。**

## 2　匹配性体检（全部满足）

| 项 | vLLM 0.29.0 要求 | 容器实际 |
|---|---|---|
| torch | `==2.13.0` | 2.13.0+cu130 ✓ |
| transformers | `>=5.10.4` | 5.15.0 ✓ |
| Python | `>=3.10,<3.15` | 3.10.20 ✓ |
| wheel 平台 | `manylinux_2_28_x86_64`（glibc≥2.28） | Ubuntu 20.04，glibc **2.31** ✓ |
| ABI | `cp38-abi3` | cp310 可用 ✓ |
| 驱动 | — | 590.48.01 ✓ |
| 磁盘 | — | 余 192G ✓ |

## 3　执行步骤（可复用）

1. **本地**（有外网）下载依赖闭包：
   ```
   pip download "vllm==0.29.0" -r <reqs.txt> -d A:\wheels_vllm \
     --only-binary=:all: --python-version 310 --implementation cp \
     --abi cp310 --abi abi3 \
     --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
     --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64
   ```
   → 175 个 wheel / 1.8G。
2. **筛掉容器已有的包**：排除 `torch-2.13.0*`、`transformers-5.17.0*`（避免覆盖容器版本）→ 173 个 / 1.3G。
3. **推送**：`scp *.whl root@<HOST>:<WHEEL_DIR>/`（约 2m13s）。
4. **建独立 venv**（关键：`--system-site-packages` 让它复用容器已装好的 torch/transformers）：
   ```
   <HOME>/dev/miniforge3/envs/safespec/bin/python -m venv <VLLM_ENV> --system-site-packages
   ```
5. **安装**：
   ```
   <VLLM_ENV>/bin/pip install --no-index --no-deps \
       --find-links <WHEEL_DIR> -r <HOME>/reqs_dedup.txt
   ```
6. **验证**：`<VLLM_ENV>/bin/python -c "import vllm; print(vllm.__version__)"` → `0.29.0` ✓

## 4　踩到的两个坑（**下次直接绕开**）

### 坑 1：pip 按**本机**平台评估环境标记，漏下 Linux 专属依赖
Windows 上执行 `pip download` 时，pip 用**当前解释器**评估 `platform_machine` / `platform_system` 标记，于是
`xgrammar`、`llguidance`、`tokenspeed-mla` 被判定"不匹配本环境"而**静默跳过**，服务端安装时报
`ERROR: Could not find a version that satisfies the requirement xgrammar...`。
**绕法**：单独按包名补下（不带标记即不评估标记），并加 `--no-deps` 防止拖出整棵依赖树：
```
pip download xgrammar llguidance tokenspeed-mla --no-deps -d <dir> --only-binary=:all: \
  --python-version 310 --implementation cp --abi cp310 --abi abi3 --platform manylinux_2_28_x86_64
```

### 坑 2：`tokenspeed-mla` 依赖一个**私有 fork**
其要求 `tokenspeed-triton>=3.7.10.post20260531`，PyPI 上不存在（`post<日期>` 后缀说明来自私有索引）。
**绕法**：改用**暴力装法**——把已下好的 wheel 按**包名去重**后整体 `--no-deps` 安装（清单 178 项），
跳过这一个可选内核包：
```
cd <WHEEL_DIR> && ls *.whl | grep -vi tokenspeed_mla | <去重> > <HOME>/reqs_dedup.txt
<VLLM_ENV>/bin/pip install --no-index --no-deps --find-links <WHEEL_DIR> -r <HOME>/reqs_dedup.txt
```
> 注意：**去重是必需的**——同一包存在多个版本时，即使加了 `--no-deps`，pip 仍会报 `ResolutionImpossible`。

## 5　后续注意事项（做对比实验时）

- **共享 GPU**：vLLM 默认 `gpu_memory_utilization=0.9`，会**预分配整卡 90% 显存**。本机与其他同事任务共用 A100-80G，必须显式压低（如 `0.10`～`0.35`）并先检查 `nvidia-smi` 空闲显存，否则会挤爆他人任务。
- **计时可比性**：vLLM 有引擎级 warmup 与 KV cache 预分配，必须固定 `gpu_memory_utilization`、`enforce_eager`、`max_model_len` 等，且与自研实现采用同一 prefill 扣除口径，才可与本文结果横向比较。
- **`enforce_eager=True`** 可绕过部分内核编译问题（容器 gcc 9.4 对 C++20 支持有限）。
