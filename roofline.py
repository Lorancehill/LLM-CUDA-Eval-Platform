"""Counter-free roofline 瓶颈指纹（不依赖 ncu / GPU 硬件计数器）。

当 ncu 不可用（WSL 权限未开 / 版本不认 GB206 / 容器内加载不到驱动 profiling 库）时，
用「实测墙钟耗时 + 算子规格 + 源码静态特征」做 roofline 判据，把内核分类为
memory / compute / latency 三类瓶颈并生成优化反馈。

相比 ncu 粗（访存字节是估算的），但跨机器可复现、零权限依赖，是「知识可迁移」的研究点：
计数器指纹在 A100 上能采、在消费卡上采不到，跨卡迁移就会断，counter-free 指纹不会。

所有函数均为纯函数，可离线单测。
"""

from __future__ import annotations

# ---- GB206 / RTX 5060 峰值参数（需按实际卡标定）----
# FP32 峰值：3840 CUDA core × 2 (FMA) × ~2.52 GHz ≈ 19.4 TFLOPS
PEAK_FLOPS_FP32 = 19.4e12        # FLOP/s
# 显存带宽：8GB GDDR7 / 128-bit / 28 Gbps ≈ 448 GB/s
PEAK_BANDWIDTH_GBS = 448.0       # GB/s

# 算力/带宽利用率都低于此值才判 latency（并行度不足，未触及任何上限）。
# 值取得很低（0.5%）：小尺寸 matmul 相对峰值的绝对利用率天然很低，阈值过高会把
# 一切判成 latency；真正的 latency 只能靠 ncu 的 stall reasons 精确判别，proxy 只兜底极端情况。
LOW_UTILIZATION = 0.005


def _clamp01(x: float) -> float:
    """把利用率夹到 [0, 1]。自报计时偏快或峰值参数偏保守时，roofline 兜底判据的
    利用率可能 >100%（如 207%），物理上不可能，落盘/反馈前夹到 1.0（P1）。"""
    if x < 0.0:
        return 0.0
    return 1.0 if x > 1.0 else x


def matmul_flops(N: int) -> float:
    """矩阵乘法总 FLOPs = 2*N^3（N*N 输出，每输出 N 次乘加）。"""
    return 2.0 * N * N * N


def estimate_traffic_bytes(N: int, has_shared_mem: bool) -> float:
    """估算访存字节（float32）。

    - 有 __shared__（分块复用）：A/B 各读一次、C 写一次，≈ 3*N^2*4（忽略 tile 边界重复的下界）。
    - 无 __shared__（naive）：每个输出线程流式读 2N 个 float（无复用）+ C 写一次，
      ≈ 2*N^3*4 + N^2*4。

    这是粗估，只用于 roofline 判据的「带宽」一侧，不追求精确。
    """
    if has_shared_mem:
        return 3.0 * N * N * 4
    return 2.0 * N * N * N * 4 + N * N * 4


def source_features(code: str | None) -> dict[str, bool]:
    """源码静态特征（辅助证据 + 反馈措辞）。"""
    code = code or ""
    return {
        "has_shared_mem": "__shared__" in code,
        "has_vectorized_load": ("float4" in code) or ("double4" in code),
        "has_restrict": "__restrict__" in code,
    }


def _feedback_for(label: str, m: dict, feat: dict[str, bool]) -> str:
    peak_gflops = PEAK_FLOPS_FP32 / 1e9
    if label == "memory":
        hint = (
            "已用 __shared__ 分块，但带宽仍受限，可再增大 tile、减少 bank conflict"
            if feat["has_shared_mem"]
            else "未使用 __shared__（每个线程反复从全局内存加载数据）"
        )
        return (
            "你的代码运行结果正确，但经 roofline 指纹（counter-free，非 ncu）判断存在**内存瓶颈**：\n"
            f"- 实测算力: {m['achieved_gflops']:.2f} GFLOP/s（峰值 {peak_gflops:.0f} GFLOP/s 的 {m['compute_utilization']*100:.2f}%）\n"
            f"- 实测带宽: {m['achieved_gbs']:.1f} GB/s（峰值 {PEAK_BANDWIDTH_GBS:.0f} GB/s 的 {m['memory_utilization']*100:.2f}%）\n"
            f"- 算术强度: {m['arithmetic_intensity']:.2f} FLOP/byte（roofline 拐点 {m['ridge_point']:.1f}）\n"
            f"- 源码特征: 共享内存={feat['has_shared_mem']}，向量化={feat['has_vectorized_load']}\n\n"
            f"带宽利用率显著高于算力利用率，说明受限于全局内存访问（{hint}）。\n"
            "请优化：用 __shared__ 做 Tiling 分块、按 tile 复用数据，减少全局内存重复访问；"
            "配合 float4 向量化访存并合理设置 block_size。重新提供完整可编译运行的 CUDA 代码。"
        )
    if label == "compute":
        return (
            "你的代码运行结果正确，但经 roofline 指纹（counter-free，非 ncu）判断存在**计算瓶颈**：\n"
            f"- 实测算力: {m['achieved_gflops']:.2f} GFLOP/s（峰值 {peak_gflops:.0f} GFLOP/s 的 {m['compute_utilization']*100:.2f}%）\n"
            f"- 实测带宽: {m['achieved_gbs']:.1f} GB/s（峰值 {PEAK_BANDWIDTH_GBS:.0f} GB/s 的 {m['memory_utilization']*100:.2f}%）\n"
            f"- 算术强度: {m['arithmetic_intensity']:.2f} FLOP/byte（roofline 拐点 {m['ridge_point']:.1f}）\n\n"
            "算力利用率高于带宽利用率，主要受计算吞吐限制。\n"
            "请优化：减少冗余计算、用 float4 向量化加载、确保内层循环可向量化为 FMA，"
            "并适当提高 block_size 提升数据复用。重新提供完整可编译运行的 CUDA 代码。"
        )
    return (
        "你的代码运行结果正确，但经 roofline 指纹（counter-free，非 ncu）判断存在**延迟/低占用瓶颈**：\n"
        f"- 实测算力: {m['achieved_gflops']:.2f} GFLOP/s（峰值 {peak_gflops:.0f} GFLOP/s 的 {m['compute_utilization']*100:.2f}%）\n"
        f"- 实测带宽: {m['achieved_gbs']:.1f} GB/s（峰值 {PEAK_BANDWIDTH_GBS:.0f} GB/s 的 {m['memory_utilization']*100:.2f}%）\n\n"
        "算力与带宽利用率都偏低，说明并行度不足、访存延迟未被隐藏。\n"
        "请优化：增大 grid 规模让更多 block 常驻 SM、适当提高 block_size、消除串行依赖与分支，"
        "提高 occupancy 以隐藏访存延迟。重新提供完整可编译运行的 CUDA 代码。"
    )


def roofline_classify(N: int, time_sec: float | None, code: str | None = None):
    """用 roofline 判据分类瓶颈，返回 (label, feedback, metrics)。

    判据：
    - 无有效计时（None / <=0）→ ('unknown', 说明, {})
    - memory  ：带宽利用率 >= 算力利用率，且带宽利用率 >= LOW_UTILIZATION
    - compute ：算力利用率 > 带宽利用率，且算力利用率 >= LOW_UTILIZATION
    - latency ：两者都低于 LOW_UTILIZATION（未触及任何上限 → 并行度不足）

    metrics 内含 profile_source='roofline_proxy' 与 timing_source='self_reported'，
    供证据链落盘区分来源（与 ncu 的 gpu__time_duration 可信计时区分开）。
    """
    if not time_sec or time_sec <= 0:
        return "unknown", "无可用计时，无法做 roofline 指纹。", {}

    flops = matmul_flops(N)
    feat = source_features(code)
    traffic_bytes = estimate_traffic_bytes(N, feat["has_shared_mem"])

    achieved_flops = flops / time_sec                # FLOP/s
    achieved_bytes = traffic_bytes / time_sec        # byte/s
    achieved_gflops = achieved_flops / 1e9
    achieved_gbs = achieved_bytes / 1e9

    compute_utilization = achieved_flops / PEAK_FLOPS_FP32
    memory_utilization = achieved_bytes / (PEAK_BANDWIDTH_GBS * 1e9)
    arithmetic_intensity = flops / traffic_bytes                 # FLOP/byte
    ridge_point = PEAK_FLOPS_FP32 / (PEAK_BANDWIDTH_GBS * 1e9)   # FLOP/byte

    if memory_utilization < LOW_UTILIZATION and compute_utilization < LOW_UTILIZATION:
        label = "latency"
    elif memory_utilization >= compute_utilization:
        label = "memory"
    else:
        label = "compute"

    # P1：兜底判据的利用率可能 >100%（自报计时偏快/峰值偏保守），夹到 [0,1] 再落盘/反馈。
    compute_utilization = _clamp01(compute_utilization)
    memory_utilization = _clamp01(memory_utilization)

    m = {
        "profile_source": "roofline_proxy",
        "timing_source": "self_reported",
        "flops": flops,
        "traffic_bytes": traffic_bytes,
        "achieved_gflops": achieved_gflops,
        "achieved_gbs": achieved_gbs,
        "compute_utilization": compute_utilization,
        "memory_utilization": memory_utilization,
        "arithmetic_intensity": arithmetic_intensity,
        "ridge_point": ridge_point,
        "has_shared_mem": feat["has_shared_mem"],
        "has_vectorized_load": feat["has_vectorized_load"],
        "has_restrict": feat["has_restrict"],
    }
    return label, _feedback_for(label, m, feat), m


def format_proxy_metrics(m: dict) -> str:
    """把 roofline 指纹格式化为单行可读文本（缺失/空返回说明）。"""
    if not m:
        return "无有效计时，无法做 roofline 指纹。"
    cu = _clamp01(float(m.get("compute_utilization", 0.0) or 0.0))
    mu = _clamp01(float(m.get("memory_utilization", 0.0) or 0.0))
    return (
        f"实测算力 {m['achieved_gflops']:.2f} GFLOP/s（{cu*100:.2f}%） | "
        f"实测带宽 {m['achieved_gbs']:.1f} GB/s（{mu*100:.2f}%） | "
        f"算术强度 {m['arithmetic_intensity']:.2f} FLOP/byte | 拐点 {m['ridge_point']:.1f} FLOP/byte"
    )
