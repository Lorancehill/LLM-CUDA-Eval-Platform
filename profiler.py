"""
Nsight Compute (ncu) 硬件性能画像模块。

在宿主侧（WSL2）直接运行 ncu，profile 容器编译出的二进制，抓取底层硬件指标并做瓶颈分类：
- 显存(DRAM)带宽利用率
- L2 Cache 命中率
- SM 计算单元吞吐率（计算占用率）
- 实际占用率 (achieved occupancy)

基于这些指标，把性能画像从「只看总耗时」升级为「内存/计算/延迟瓶颈」分类，
并生成可直接喂回大模型的优化反馈。

本模块不依赖 GPU：命令构造、CSV 解析、瓶颈分类均为纯函数，可离线单元测试。

健壮性约定（v2.1）：解析入口容忍 None / bytes / 畸形表头与带 % 单位的值；指标统一归一化成
有限 float（NaN/Inf/字符串非法值视为缺失，不参与比较）；重复指标取首次出现，保证结果确定；
格式化对缺失/坏值显示 N/A；含空格路径在命令中按需引用，避免 async_eval 的 shlex.split 断参。
"""

from __future__ import annotations

import csv
import glob
import io
import math
import os
import re
import shlex
import shutil
from typing import Any

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

# 指标名常量：避免多处手写完整字符串，也方便测试断言（顺序与 METRICS 一致）。
_M_TIME = "gpu__time_duration.sum"
_M_DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
_M_L2 = "lts__t_sector_hit_rate.pct"
_M_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
_M_OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"


def _coerce_text(value: object) -> str :
    """
    把任意输入统一成 str（解析入口的防御层）。

    None → ""；bytes/bytearray → UTF-8 解码（非法字节用 replacement 兜底）；
    其它类型 → str(value)。
    """
    if value is None :
        return ""
    if isinstance(value, (bytes, bytearray)) :
        return bytes(value).decode("utf-8", errors = "replace")
    return value if isinstance(value, str) else str(value)


def _header_cell(value: str) -> str :
    """
    归一化 CSV 表头单元格：去空格、BOM、被当成普通字符的引号，再转小写。

    ncu 输出若被包了一层（BOM 出现在引号前），csv 不会把该字段识别为 quoted field，
    引号会留在值里；这里统一剥掉，保证仍能按列名定位。
    """
    return value.strip().lstrip("\ufeff").strip().strip('"').strip().lower()


def _parse_float(value: object, *, percent_ok: bool = False) -> float | None :
    """
    把指标值归一化成有限 float；None / 不可转换 / NaN / ±Inf → None。

    percent_ok=True 时容忍 ncu 某些输出把单位拼进值里的写法（"78.5%" / "30.0 %"）。
    """
    if value is None :
        return None
    if isinstance(value, str) :
        text = value.strip()
        if percent_ok and text.endswith("%") :
            text = text[:-1].strip()
        try :
            number = float(text)
        except (TypeError, ValueError, OverflowError) :
            return None
    else :
        try :
            number = float(value)
        except (TypeError, ValueError, OverflowError) :
            return None
    return number if math.isfinite(number) else None


def _quote_if_needed(token: str) -> str :
    """
    含空白字符的路径/参数按需 shlex 引用；普通 token 原样返回（保持命令字符串兼容）。
    """
    return shlex.quote(token) if any(char.isspace() for char in token) else token


def _ncu_version_key(path: str) -> tuple[int, ...] :
    """
    从 ~/.local/nsight-compute/<ver>/ncu 提取版本号元组，供按版本数值排序。

    直接字符串排序会把 "2026.10" 排到 "2026.4" 后面（'1' < '4'），选错旧版；
    按元组比较才是正确语义（2026,10）>（2026,4）。
    """
    m = re.search(r"nsight-compute[/\\]([0-9]+(?:\.[0-9]+)*)[/\\]", path or "")
    if not m :
        return ()
    return tuple(int(x) for x in m.group(1).split("."))


def list_ncu_candidates(env: dict[str, str] | None = None) -> list[str] :
    """
    返回候选 ncu 二进制路径（按优先级，纯函数、不跑子进程）。

    新版 ncu（支持 gb206 的 2026.x）常装在 ``~/.local/nsight-compute/*/ncu`` 且**不进
    PATH**——普通 shell 里的 ``ncu`` 仍是旧的 /usr/bin/ncu（2024.1.1，不支持 gb206），
    直接跑会静默退化到 roofline（见论文可行性清单 3.1.1）。

    优先级：
    1. ``NCU_BIN`` 环境变量（显式指定，最高优先级）；
    2. ``~/.local/nsight-compute/*/ncu``（自装新版，按目录版本号新→旧排序）；
    3. ``shutil.which("ncu")``（PATH 中的 ncu，通常是系统旧版）；
    4. 兜底字符串 ``"ncu"``（交给 subprocess 报 FileNotFoundError，上层优雅降级）。

    结果保序去重（NCU_BIN 与 glob/which 指向同一路径时不会重复探测）。
    ``env`` 参数供测试注入；缺省取 ``os.environ``。
    """
    env = os.environ if env is None else env
    explicit = (env.get("NCU_BIN") or "").strip()
    home = os.path.expanduser("~")
    pattern = os.path.join(home, ".local", "nsight-compute", "*", "ncu")
    # 按版本号数值新→旧排序（字符串排序会把 2026.10 排到 2026.4 之后，见 _ncu_version_key）。
    installed = sorted(glob.glob(pattern), key = _ncu_version_key, reverse = True)
    which = shutil.which("ncu")

    ordered = [path for path in [explicit, *installed, which, "ncu"] if path]
    return list(dict.fromkeys(ordered))


def build_ncu_cmd(binary: str, kernel_name: str | None = None, ncu_bin: str = "ncu") -> str :
    """
    构造在宿主侧运行的 ncu 命令行（默认只 profile 第一次 kernel launch，输出 CSV）。

    kernel_name 传入**正则片段**（不含 ``regex:`` 前缀），用于在多 kernel 代码中只
    profile 目标内核（否则 --launch-count 1 会误采第一个 kernel，如 init kernel /
    融合算子）；为 None 时保持「第一个 kernel launch」的默认行为。

    宿主侧（WSL2）适配：
    - ``--clock-control none``：不锁频，避免 WSL 下无权限修改时钟；
    - ``--target-processes all``：允许注入到全部相关进程；
    - ``--kernel-name-base demangled``：按 demangled 内核名匹配正则。

    含空格的 ncu_bin / binary / kernel 正则会按需引用，保证 async_eval 的 shlex.split
    能还原成单个参数；普通输入的命令字符串保持不变。
    """
    base = (
        f"{_quote_if_needed(ncu_bin)} --csv --print-metric-name name "
        "--clock-control none --target-processes all "
    )
    binary_arg = _quote_if_needed(f"./{binary}")
    metrics = ",".join(METRICS)
    if kernel_name :
        # 按内核名（demangled 正则）过滤，只 profile 命中的第一个 launch
        return (
            base
            + "--kernel-name-base demangled "
            + f"--kernel-name {_quote_if_needed(f'regex:{kernel_name}')} "
            + "--launch-count 1 --launch-skip 0 "
            + f"--metrics {metrics} {binary_arg}"
        )
    return (
        base
        + "--launch-count 1 --launch-skip 0 "
        + f"--metrics {metrics} {binary_arg}"
    )


def parse_ncu_csv(stdout: str | bytes | None) -> dict[str, float] :
    """
    解析 ncu --csv 输出，返回 {metric_name: value}。

    解析是「表头驱动」的：先找含 ``Metric Name`` / ``Metric Value`` 的表头行定位列，
    再按列名取值。这样兼容不同 ncu 版本的列顺序与多出的列；同时跳过 app 自身
    混入 stdout 的非 CSV 行（如 ``Time: ...``），只按指标名精确匹配。

    健壮性：None/bytes 输入、表头 BOM/首尾空格、值带 ``%`` 等单位、同一指标重复出现
    （多 kernel/多 section）都按约定处理——重复时**首次出现优先**，结果确定可复现。
    """
    result: dict[str, float] = {}
    reader = csv.reader(io.StringIO(_coerce_text(stdout)))
    name_idx: int | None = None
    value_idx: int | None = None

    for row in reader :
        if not row :
            continue
        if name_idx is None :
            # 表头可能带 BOM / 首尾空格；只按列名定位，不依赖固定下标。
            lowered = [_header_cell(cell) for cell in row]
            if "metric name" in lowered and "metric value" in lowered :
                name_idx = lowered.index("metric name")
                value_idx = lowered.index("metric value")
            continue
        if len(row) <= max(name_idx, value_idx) :
            continue
        name = row[name_idx].strip()
        if name in METRICS and name not in result :
            value = _parse_float(row[value_idx], percent_ok = True)
            if value is not None :
                result[name] = value
    return result


def format_metrics(m: dict[str, Any] | None) -> str :
    """
    把指标字典格式化为可读的多行文本（缺失/坏值显示 N/A）。
    """
    metrics = m if isinstance(m, dict) else {}
    lines = []
    for name in METRICS :
        label = METRIC_LABELS[name]
        value = _parse_float(metrics.get(name))
        if value is None :
            lines.append(f"  {label}: N/A")
        elif name == _M_TIME :
            lines.append(f"  {label}: {value / 1e6:.3f} ms")
        else :
            lines.append(f"  {label}: {value:.1f}%")
    return "\n".join(lines)


def _fmt_pct(m: dict[str, Any], name: str) -> str :
    value = _parse_float(m.get(name))
    return "N/A" if value is None else f"{value:.1f}%"


def classify_bottleneck(m: dict[str, Any] | None) -> tuple[str, str] :
    """
    根据硬件指标做瓶颈分类，返回 (标签, 反馈文本)。

    标签取值：
    - 'memory'   ：显存带宽利用率高但 SM 占用率低 → 内存瓶颈（喂回 AI 让其优化）
    - 'compute'  ：SM 占用率高 → 计算瓶颈
    - 'latency'  ：两者都低 → 延迟/低占用瓶颈
    - 'unknown'  ：关键指标缺失或非有限，无法分类
    """
    metrics = m if isinstance(m, dict) else {}
    sm = _parse_float(metrics.get(_M_SM))
    dram = _parse_float(metrics.get(_M_DRAM))

    if sm is None or dram is None :
        return "unknown", "关键指标缺失，无法分类。"

    if dram >= DRAM_BOUND_PCT and sm < SM_SATURATED_PCT :
        l2 = _fmt_pct(metrics, _M_L2)
        occ = _fmt_pct(metrics, _M_OCC)
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

    if sm >= SM_SATURATED_PCT :
        occ = _fmt_pct(metrics, _M_OCC)
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

    occ = _fmt_pct(metrics, _M_OCC)
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
