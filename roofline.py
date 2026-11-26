"""
Counter-free roofline 瓶颈指纹（不依赖 ncu / GPU 硬件计数器）。

当 ncu 不可用（WSL 权限未开 / 版本不认 GB206 / 容器内加载不到驱动 profiling 库）时，
用「实测墙钟耗时 + 算子规格 + 源码静态特征」做 roofline 判据，把内核分类为
memory / compute / latency 三类瓶颈并生成优化反馈。

相比 ncu 粗（访存字节是估算的），但跨机器可复现、零权限依赖，是「知识可迁移」的研究点：
计数器指纹在 A100 上能采、在消费卡上采不到，跨卡迁移就会断，counter-free 指纹不会。

所有函数均为纯函数，可离线单测。

健壮性约定（v2.1）：矩阵尺寸无效或耗时非有限/非正时统一返回 unknown，不抛异常、不产生
NaN 指标；源码特征匹配前先剥注释与字符串字面量，避免注释里的关键字污染 RAG 归因；
format_proxy_metrics 对缺失键容错，坏数据不会让看板/日志崩溃。

额外提供 roofline_efficiency(N, time_sec)：用「理想下界（算力/带宽取 max）÷ 实测耗时」给出
相对 roofline 的达成率（1.0 = 达到理论上限，0.3 = 达到 30%），并与 metrics 一起落盘。
"""

from __future__ import annotations

import math
import re
from typing import Any

# ---- RTX 5060 Laptop / GB206（sm_120）峰值参数 ----
# 来源：容器内 deviceQuery（CUDA 12.8, sm_120）实测 + 规格推算，勿再用桌面 5060 的数值。
#   deviceQuery: multiProcessorCount=26, clockRate=1687000kHz(默认/基频), warpSize=32,
#                memoryClockRate=12001000kHz, memoryBusWidth=128, sharedMemPerMultiprocessor=102400B
#   FP32 峰值 ≈ 26 SM × 128 FP32 core/SM × 2 (FMA) × 1.687 GHz ≈ 11.2 TFLOPS（基频保守值）
#   显存带宽 ≈ 12001 MHz × 2 (DDR) × 128-bit / 8 ≈ 384 GB/s
# 如需更精确，可锁频后用纯 FMA kernel / copy kernel 实测峰值并替换这两个常数。
PEAK_FLOPS_FP32 = 11.2e12        # FLOP/s（deviceQuery 基频推算）
PEAK_BANDWIDTH_GBS = 384.0       # GB/s（deviceQuery 显存频率推算）

# 算力/带宽利用率都低于此值才判 latency（并行度不足，未触及任何上限）。
# 值取得很低（0.5%）：小尺寸 matmul 相对峰值的绝对利用率天然很低，阈值过高会把
# 一切判成 latency；真正的 latency 只能靠 ncu 的 stall reasons 精确判别，proxy 只兜底极端情况。
LOW_UTILIZATION = 0.005

# 源码静态特征匹配前先剥掉注释与字符串字面量：注释里出现 "__shared__" / "float4"
# 不应被当成真实实现（否则会污染 RAG 的 applied_cards 归因与候选卡 fingerprint）。
_COMMENT_OR_STRING_RE: re.Pattern[str] = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)

# 向量化访存类型：float4/double4 之外也覆盖 float2/int4/half4 等常见写法。
_VECTORIZED_LOAD_RE: re.Pattern[str] = re.compile(
    r"\b(?:float|double|half|int|uint|short|char)[248]\b"
)


def _coerce_code(code: object) -> str :
    """
    把源码输入统一成 str（容忍 None / bytes / 其它类型），供特征匹配使用。
    """
    if code is None :
        return ""
    if isinstance(code, (bytes, bytearray)) :
        return bytes(code).decode("utf-8", errors = "replace")
    return code if isinstance(code, str) else str(code)


def _strip_comments_and_strings(code: str) -> str :
    """
    去掉 C/C++ 注释与字符串字面量，只保留真正的代码文本。
    """
    return _COMMENT_OR_STRING_RE.sub("", code)


def _to_finite_seconds(value: object) -> float | None :
    """
    把耗时归一化成「有限正 float」；None / 非正 / NaN / ±Inf / 不可转换 → None。
    """
    if value is None :
        return None
    try :
        seconds = float(value)
    except (TypeError, ValueError, OverflowError) :
        return None
    if not math.isfinite(seconds) or seconds <= 0.0 :
        return None
    return seconds


def _normalize_size(N: object) -> int | float | None :
    """
    把矩阵尺寸归一化成「有限正数」；不可转换 / 非正 / NaN / ±Inf / bool → None。

    返回 int（整数尺寸）或 float（如 1024.5），供 matmul_flops / 流量估算使用；
    先归一化再计算，避免 classify("1024", ...) 这种输入在乘法处 TypeError。
    """
    if N is None or isinstance(N, bool) :
        return None
    try :
        size = float(N)
    except (TypeError, ValueError, OverflowError) :
        return None
    if not math.isfinite(size) or size <= 0.0 :
        return None
    return int(size) if size.is_integer() else size


def _clamp01(x: float) -> float :
    """
    把利用率夹到 [0, 1]。自报计时偏快或峰值参数偏保守时，roofline 兜底判据的
    利用率可能 >100%（如 207%），物理上不可能，落盘/反馈前夹到 1.0（P1）。
    """
    if not math.isfinite(x) or x < 0.0 :
        return 0.0
    return min(x, 1.0)


def matmul_flops(N: int | float) -> float :
    """
    矩阵乘法总 FLOPs = 2*N^3（N*N 输出，每输出 N 次乘加）。
    """
    return 2.0 * N * N * N


def estimate_traffic_bytes(N: int | float, has_shared_mem: bool) -> float :
    """
    估算访存字节（float32）。

    - 有 __shared__（分块复用）：A/B 各读一次、C 写一次，≈ 3*N^2*4（忽略 tile 边界重复的下界）。
    - 无 __shared__（naive）：每个输出线程流式读 2N 个 float（无复用）+ C 写一次，
      ≈ 2*N^3*4 + N^2*4。

    这是粗估，只用于 roofline 判据的「带宽」一侧，不追求精确。
    """
    if has_shared_mem :
        return 3.0 * N * N * 4
    return 2.0 * N * N * N * 4 + N * N * 4


def _ideal_traffic_bytes(N: int | float) -> float :
    """
    matmul 的理想最小全局访存字节（float32）：读 A、B 各一次 + 写 C 一次 = 3*N^2*4。

    与 estimate_traffic_bytes 的区别：后者按"实现方式"估实际流量（naive 会远大于理想值），
    用于判瓶颈；本函数是任何实现都无法低于的理论下界，用于算 roofline 效率。
    """
    return 3.0 * N * N * 4


def source_features(code: str | bytes | None) -> dict[str, bool] :
    """
    源码静态特征（辅助证据 + 反馈措辞）。

    匹配前先剥注释与字符串字面量；has_vectorized_load 覆盖 float4/double4/float2/int4 等。
    """
    clean_code = _strip_comments_and_strings(_coerce_code(code))
    return {
        "has_shared_mem": "__shared__" in clean_code,
        "has_vectorized_load": bool(_VECTORIZED_LOAD_RE.search(clean_code)),
        "has_restrict": "__restrict__" in clean_code,
    }


def _feedback_for(label: str, m: dict[str, Any], feat: dict[str, bool]) -> str :
    peak_gflops = PEAK_FLOPS_FP32 / 1e9
    if label == "memory" :
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
    if label == "compute" :
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


def roofline_classify(N: object, time_sec: float | None, code: str | None = None) -> tuple[str, str, dict[str, Any]] :
    """
    用 roofline 判据分类瓶颈，返回 (label, feedback, metrics)。

    判据：
    - 无有效计时（None / <=0 / NaN / Inf）或尺寸无效 → ('unknown', 说明, {})
    - memory  ：带宽利用率 >= 算力利用率，且带宽利用率 >= LOW_UTILIZATION
    - compute ：算力利用率 > 带宽利用率，且算力利用率 >= LOW_UTILIZATION
    - latency ：两者都低于 LOW_UTILIZATION（未触及任何上限 → 并行度不足）

    metrics 内含 profile_source='roofline_proxy' 与 timing_source='self_reported'，
    供证据链落盘区分来源（与 ncu 的 gpu__time_duration 可信计时区分开）。
    """
    seconds = _to_finite_seconds(time_sec)
    if seconds is None :
        return "unknown", "无可用计时，无法做 roofline 指纹。", {}
    size = _normalize_size(N)
    if size is None :
        return "unknown", "无效的矩阵尺寸，无法做 roofline 指纹。", {}

    flops = matmul_flops(size)
    feat = source_features(code)
    traffic_bytes = estimate_traffic_bytes(size, feat["has_shared_mem"])

    achieved_flops = flops / seconds                # FLOP/s
    achieved_bytes = traffic_bytes / seconds        # byte/s
    achieved_gflops = achieved_flops / 1e9
    achieved_gbs = achieved_bytes / 1e9

    compute_utilization = achieved_flops / PEAK_FLOPS_FP32
    memory_utilization = achieved_bytes / (PEAK_BANDWIDTH_GBS * 1e9)
    arithmetic_intensity = flops / traffic_bytes                 # FLOP/byte
    ridge_point = PEAK_FLOPS_FP32 / (PEAK_BANDWIDTH_GBS * 1e9)   # FLOP/byte

    if memory_utilization < LOW_UTILIZATION and compute_utilization < LOW_UTILIZATION :
        label = "latency"
    elif memory_utilization >= compute_utilization :
        label = "memory"
    else :
        label = "compute"

    # P1：兜底判据的利用率可能 >100%（自报计时偏快/峰值偏保守），夹到 [0,1] 再落盘/反馈。
    compute_utilization = _clamp01(compute_utilization)
    memory_utilization = _clamp01(memory_utilization)

    # roofline 效率：理想下界（算力/带宽取 max）÷ 实测耗时；与实现无关，回答"离理论上限多远"。
    ideal_traffic_bytes = _ideal_traffic_bytes(size)
    roofline_bound_sec = max(
        flops / PEAK_FLOPS_FP32,
        ideal_traffic_bytes / (PEAK_BANDWIDTH_GBS * 1e9),
    )
    roofline_efficiency = _clamp01(roofline_bound_sec / seconds)

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
        "ideal_traffic_bytes": ideal_traffic_bytes,
        "roofline_bound_sec": roofline_bound_sec,
        "roofline_efficiency": roofline_efficiency,
    }
    return label, _feedback_for(label, m, feat), m


def roofline_efficiency(N: object, time_sec: float | None) -> float | None :
    """
    roofline 效率 = 理想下界耗时 / 实测耗时（1.0 = 达到理论上限，0.3 = 30%）。

    理想下界 = max(2N^3 / PEAK_FLOPS_FP32, 3N^2*4 / (PEAK_BANDWIDTH_GBS*1e9))：
    - 算力侧：FLOPs 上限决定的最短时间；
    - 带宽侧：A/B 各读一次 + C 写一次（3N^2*4 字节）决定的最短时间。
    该指标与实现方式无关（naive 的重复访存不会抬高下界），因此比"实际流量利用率"更诚实。

    尺寸/耗时无效返回 None；结果夹到 [0,1]（峰值参数保守或 L2 复用时可能 >1，夹到 1.0）。
    """
    seconds = _to_finite_seconds(time_sec)
    size = _normalize_size(N)
    if seconds is None or size is None :
        return None

    flops = matmul_flops(size)
    ideal_traffic_bytes = _ideal_traffic_bytes(size)
    bound_sec = max(
        flops / PEAK_FLOPS_FP32,
        ideal_traffic_bytes / (PEAK_BANDWIDTH_GBS * 1e9),
    )
    return _clamp01(bound_sec / seconds)


def format_proxy_metrics(m: dict[str, Any]) -> str :
    """
    把 roofline 指纹格式化为单行可读文本（缺失/空返回说明，缺键按 0 兜底）。
    """
    if not m :
        return "无有效计时，无法做 roofline 指纹。"

    def _value(key: str) -> float :
        try :
            return float(m.get(key, 0.0) or 0.0)
        except (TypeError, ValueError) :
            return 0.0

    cu = _clamp01(_value("compute_utilization"))
    mu = _clamp01(_value("memory_utilization"))
    return (
        f"实测算力 {_value('achieved_gflops'):.2f} GFLOP/s（{cu*100:.2f}%） | "
        f"实测带宽 {_value('achieved_gbs'):.1f} GB/s（{mu*100:.2f}%） | "
        f"算术强度 {_value('arithmetic_intensity'):.2f} FLOP/byte | 拐点 {_value('ridge_point'):.1f} FLOP/byte"
    )


__all__ = [
    "LOW_UTILIZATION",
    "PEAK_BANDWIDTH_GBS",
    "PEAK_FLOPS_FP32",
    "estimate_traffic_bytes",
    "format_proxy_metrics",
    "matmul_flops",
    "roofline_classify",
    "roofline_efficiency",
    "source_features",
]
