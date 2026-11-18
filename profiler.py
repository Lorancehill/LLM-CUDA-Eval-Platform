"""Nsight Compute (ncu) 硬件性能画像模块。

在宿主侧（WSL2）直接运行 ncu，profile 容器编译出的二进制，抓取底层硬件指标并做瓶颈分类：
- 显存(DRAM)带宽利用率
- L2 Cache 命中率
- SM 计算单元吞吐率（计算占用率）
- 实际占用率 (achieved occupancy)

基于这些指标，把性能画像从「只看总耗时」升级为「内存/计算/延迟瓶颈」分类，
并生成可直接喂回大模型的优化反馈。

本模块不依赖 GPU：命令构造、CSV 解析、瓶颈分类均为纯函数，可离线单元测试。
"""

from __future__ import annotations

import csv
import io

# 要抓取的 ncu 指标（在 Blackwell sm_120 上稳定存在，均为 ncu 文档化指标）。
# 除 gpu__time_duration.sum（单位 ns）外，其余均为 0~100 的百分比。
METRICS = [
    "gpu__time_duration.sum",                                  # 内核耗时 (ns)
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",  # 显存(DRAM)带宽利用率 (%)
    "lts__t_sector_hit_rate.pct",                              # L2 Cache 命中率 (%)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",        # SM 计算单元吞吐/占用率 (%)
    "sm__warps_active.avg.pct_of_peak_sustained_active",       # 实际占用率 (achieved occupancy, %)
]

# 指标中文标签，用于打印与反馈
METRIC_LABELS = {
    "gpu__time_duration.sum": "内核耗时",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "显存带宽利用率",
    "lts__t_sector_hit_rate.pct": "L2 Cache 命中率",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM 计算单元占用率",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "实际占用率",
}

# 瓶颈分类阈值（百分比）
DRAM_BOUND_PCT = 60.0    # 显存带宽利用率 >= 60% 视为「显存读取很高」
SM_SATURATED_PCT = 60.0  # SM 占用率 >= 60% 视为「计算单元吃满」


def build_ncu_cmd(binary: str, kernel_name: str | None = None) -> str:
    """构造在宿主侧运行的 ncu 命令行（默认只 profile 第一次 kernel launch，输出 CSV）。

    kernel_name 传入**正则片段**（不含 ``regex:`` 前缀），用于在多 kernel 代码中只
    profile 目标内核（否则 --launch-count 1 会误采第一个 kernel，如 init kernel /
    融合算子）；为 None 时保持「第一个 kernel launch」的默认行为。

    宿主侧（WSL2）适配：
    - ``--clock-control none``：不锁频，避免 WSL 下无权限修改时钟；
    - ``--target-processes all``：允许注入到全部相关进程；
    - ``--kernel-name-base demangled``：按 demangled 内核名匹配正则。
    """
    base = (
        "ncu --csv --print-metric-name name "
        "--clock-control none --target-processes all "
    )
    if kernel_name:
        # 按内核名（demangled 正则）过滤，只 profile 命中的第一个 launch
        return (
            base
            + f"--kernel-name-base demangled --kernel-name regex:{kernel_name} "
            + "--launch-count 1 --launch-skip 0 "
            + f"--metrics {','.join(METRICS)} ./{binary}"
        )
    return (
        base
        + "--launch-count 1 --launch-skip 0 "
        + f"--metrics {','.join(METRICS)} ./{binary}"
    )


def parse_ncu_csv(stdout: str) -> dict[str, float]:
    """解析 ncu --csv 输出，返回 {metric_name: value}。

    解析是「表头驱动」的：先找含 ``Metric Name`` / ``Metric Value`` 的表头行定位列，
    再按列名取值。这样兼容不同 ncu 版本的列顺序与多出的列；同时跳过 app 自身
    混入 stdout 的非 CSV 行（如 ``Time: ...``），只按指标名精确匹配。
    """
    result: dict[str, float] = {}
    reader = csv.reader(io.StringIO(stdout))
    name_idx: int | None = None
    value_idx: int | None = None

    for row in reader:
        if not row:
            continue
        if name_idx is None:
            lowered = [c.strip().lower() for c in row]
            if "metric name" in lowered and "metric value" in lowered:
                name_idx = lowered.index("metric name")
                value_idx = lowered.index("metric value")
            continue
        if len(row) <= max(name_idx, value_idx):
            continue
        name = row[name_idx].strip()
        if name in METRICS:
            try:
                result[name] = float(row[value_idx])
            except ValueError:
                continue
    return result


def format_metrics(m: dict[str, float]) -> str:
    """把指标字典格式化为可读的多行文本（缺失指标显示 N/A）。"""
    lines = []
    for name in METRICS:
        label = METRIC_LABELS[name]
        val = m.get(name)
        if val is None:
            lines.append(f"  {label}: N/A")
        elif name == "gpu__time_duration.sum":
            lines.append(f"  {label}: {val / 1e6:.3f} ms")
        else:
            lines.append(f"  {label}: {val:.1f}%")
    return "\n".join(lines)


def _fmt_pct(m: dict[str, float], name: str) -> str:
    val = m.get(name)
    return "N/A" if val is None else f"{val:.1f}%"


def classify_bottleneck(m: dict[str, float]) -> tuple[str, str]:
    """根据硬件指标做瓶颈分类，返回 (标签, 反馈文本)。

    标签取值：
    - 'memory'   ：显存带宽利用率高但 SM 占用率低 → 内存瓶颈（喂回 AI 让其优化）
    - 'compute'  ：SM 占用率高 → 计算瓶颈
    - 'latency'  ：两者都低 → 延迟/低占用瓶颈
    - 'unknown'  ：关键指标缺失，无法分类
    """
    sm = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    dram = m.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")

    if sm is None or dram is None:
        return "unknown", "关键指标缺失，无法分类。"

    if dram >= DRAM_BOUND_PCT and sm < SM_SATURATED_PCT:
        l2 = _fmt_pct(m, "lts__t_sector_hit_rate.pct")
        occ = _fmt_pct(m, "sm__warps_active.avg.pct_of_peak_sustained_active")
        return "memory", (
            "你的代码运行结果正确，但经 Nsight Compute 硬件画像发现存在**内存瓶颈**：\n"
            f"- 显存(DRAM)带宽利用率: {dram:.1f}%\n"
            f"- SM 计算单元占用率: {sm:.1f}%\n"
            f"- L2 Cache 命中率: {l2}\n"
            f"- 实际占用率: {occ}\n\n"
            "计算单元远未吃满，但显存读取很高，说明 kernel 受限于全局内存访问带宽"
            "（典型如 naive 全局内存矩阵乘法，每个线程反复从 global memory 加载数据）。\n"
            "请优化：使用 __shared__ 共享内存做 Tiling（分块），把数据先加载进共享内存再复用，"
            "减少对全局内存的重复访问；并合理设置 block_size（如 16x16 或 32x32）。"
            "重新提供完整可编译运行的 CUDA 代码。"
        )

    if sm >= SM_SATURATED_PCT:
        occ = _fmt_pct(m, "sm__warps_active.avg.pct_of_peak_sustained_active")
        return "compute", (
            "你的代码运行结果正确，但经 Nsight Compute 硬件画像发现存在**计算瓶颈**：\n"
            f"- SM 计算单元占用率: {sm:.1f}%\n"
            f"- 实际占用率: {occ}\n"
            f"- 显存(DRAM)带宽利用率: {dram:.1f}%\n\n"
            "计算单元已基本吃满，主要受计算吞吐限制（典型如每个线程大量重复计算、"
            "未用向量化访存 / FMA 指令）。\n"
            "请优化：减少冗余计算、用 float4 向量化加载、确保内层循环可被编译器向量化为 FMA，"
            "并适当调高 block_size 提升数据复用。重新提供完整可编译运行的 CUDA 代码。"
        )

    occ = _fmt_pct(m, "sm__warps_active.avg.pct_of_peak_sustained_active")
    return "latency", (
        "你的代码运行结果正确，但经 Nsight Compute 硬件画像发现存在**延迟/低占用瓶颈**：\n"
        f"- SM 计算单元占用率: {sm:.1f}%\n"
        f"- 显存(DRAM)带宽利用率: {dram:.1f}%\n"
        f"- 实际占用率: {occ}\n\n"
        "SM 与显存利用率都偏低，说明并行度不足、访存延迟未被隐藏"
        "（典型如 block 数量太少、block_size 过小或存在串行依赖/分支）。\n"
        "请优化：增大 grid 规模让更多 block 常驻 SM、适当提高 block_size、"
        "消除 kernel 内串行依赖与分支，提高 occupancy 以隐藏访存延迟。"
        "重新提供完整可编译运行的 CUDA 代码。"
    )
