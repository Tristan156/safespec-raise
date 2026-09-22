"""safespec —— 长上下文推测解码的收益判据与增量回退调度（参考实现）。

模块：
    spec_engine   常规贪心解码 / 块验证推测解码 / 接受率驱动动态 k 基线
    raise_engine  RAISE 调度内核（M1 增量回退 + M2 置信度门控自适应深度 + M3 严格 EOS）
    data_loader   LongBench 派生的 10 个请求剖面构造

注意：``spec_engine`` / ``raise_engine`` 依赖 PyTorch，故此处不 eager import，
避免只想用 ``data_loader`` 的场景被迫安装 torch。直接按子模块导入即可：

    from safespec.spec_engine import load_model, speculative_decode
    from safespec.raise_engine import raise_decode
    from safespec.data_loader import build_profiles, load_longbench_from_dir
"""
__all__ = ["spec_engine", "raise_engine", "data_loader"]
__version__ = "1.0.0"
