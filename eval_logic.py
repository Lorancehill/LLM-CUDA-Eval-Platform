"""
评测纯逻辑：耗时解析、best-so-far 判据、结果落盘（results schema v2）。

从 async_eval 抽出，便于在**不导入 async_eval**（它 import 即创建 API client）的前提下
单测这些逻辑。函数不依赖 GPU / Docker / API。
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Literal, TypedDict


class CompileError(TypedDict):
    """一条结构化编译错误（extract_compile_errors 的返回元素）。"""

    file: str | None
    line: int | None
    severity: str
    message: str
    source: str


FailureKind = Literal["timeout", "run_failed", "no_output", "wrong_result"]

_TIME_RE: re.Pattern[str] = re.compile(r"Time:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")

# 编译器报错行（自我纠错闭环用，见 extract_compile_errors）。
# ① nvcc：matrix_mul_256.cu(12): error: identifier "foo" is undefined
# ② 宿主 g++：/tmp/.../main.cu:12:5: error: 'foo' was not declared in this scope
# ③ ptxas（编译期、无行号）：ptxas error   : Entry function ... uses too much shared data
_NVCC_ERR_RE: re.Pattern[str] = re.compile(
    r"(?P<file>[^\s(),:]+)?\s*\((?P<line>\d+)\):\s*(?P<sev>error|warning|note):\s*(?P<msg>.*)"
)
_GCC_ERR_RE: re.Pattern[str] = re.compile(
    r"(?P<file>[^\s():,]+):(?P<line>\d+)(?::\d+)?:\s*(?P<sev>fatal error|error|warning|note):\s*(?P<msg>.*)"
)
_PTXAS_ERR_RE: re.Pattern[str] = re.compile(
    r"\bptxas\s+(?P<sev>fatal|error|warning)\s*:\s*(?P<msg>.*)"
)


def parse_time_seconds(stdout_str: str | None) -> float | None :
    """
    从 CUDA 输出解析 'Time: xxx seconds' 中的秒数；无匹配返回 None。
    """
    m = _TIME_RE.search(stdout_str or "")

    return float(m.group(1)) if m else None


def parse_time_seconds_median(stdout_str: str | None) -> float | None :
    """
    解析全部 'Time: xxx seconds' 并返回中位数（论文计时口径：多次重复取中位数，
    抗首轮冷启动/偶发抖动）；无匹配返回 None。
    """
    vals = [float(m.group(1)) for m in _TIME_RE.finditer(stdout_str or "")]
    if not vals :
        return None

    vals.sort()

    n = len(vals)
    mid = n // 2

    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def should_update_best(best_ok: bool, best_time: float | None, new_time: float | None) -> bool :
    """
    best-so-far 更新判据（纯函数）。

    - 尚无正确结果（best_ok = False）：无条件记录（首个正确结果）。
    - 已有正确结果：
        - 旧耗时缺失、新耗时存在 → 更新（能测得就更可信，补齐缺失计时）；
        - 新旧都测得且新耗时严格更小 → 更新；
        - 否则（无法证明更快）保留旧 best，避免被改坏的内核「反杀」。
    """
    if not best_ok :
        return True
    if best_time is None :
        return new_time is not None
    return new_time is not None and new_time < best_time


def sanitize_float(v: float | None) -> float | None :
    """
    非有限值（NaN/Inf）-> None，其余原样。

    json.dump 默认会写出非法 JSON 的 NaN/Infinity；落盘前统一清洗。
    无法转成 float 的输入原样返回（调用方语义不变）。
    """
    if v is None :
        return None
    try :
        f = float(v)
    except (TypeError, ValueError) :
        return v
    return None if not math.isfinite(f) else v


def compute_speedup(prev_time: float | int | None, cur_time: float | int | None) -> float | None :
    """
    加速比 = prev_time / cur_time（>1 更快，<1 更慢，==1 持平）。

    两者都必须是正的有限数才可比较；否则返回 None（无法比较）。
    用于记录「本轮相对上一成功轮的提速」，与 best-so-far 的绝对保优判据互补。
    """
    try :
        p = float(prev_time)
        c = float(cur_time)
    except (TypeError, ValueError) :
        return None
    if not (math.isfinite(p) and math.isfinite(c)) :   # NaN / ±Inf
        return None
    if p <= 0 or c <= 0 :
        return None
    return p / c


def extract_compile_errors(text: str | None) -> list[CompileError] :
    """
    从 nvcc / g++ / ptxas 编译输出里提取结构化报错行（file/line/severity/message）。

    自我纠错闭环用：把「具体行号 + 报错原因」喂回模型，而不是只给 stderr 尾部 300 字符。
    覆盖三种形态（示例见 _NVCC_ERR_RE / _GCC_ERR_RE / _PTXAS_ERR_RE）：
    - nvcc："matrix_mul_256.cu(12): error: identifier "foo" is undefined"
    - 宿主 g++："/tmp/.../main.cu:12:5: error: 'foo' was not declared"
    - ptxas（编译期、无行号）："ptxas error : Entry function ... uses too much shared data"
    无匹配返回空列表。
    """
    out: list[CompileError] = []
    for m in _NVCC_ERR_RE.finditer(text or "") :
        out.append({
            "file": m.group("file") or None,
            "line": int(m.group("line")),
            "severity": m.group("sev"),
            "message": (m.group("msg") or "").strip(),
            "source": "nvcc",
        })
    for m in _GCC_ERR_RE.finditer(text or "") :
        out.append({
            "file": m.group("file") or None,
            "line": int(m.group("line")),
            "severity": m.group("sev"),
            "message": (m.group("msg") or "").strip(),
            "source": "gcc",
        })
    for m in _PTXAS_ERR_RE.finditer(text or "") :
        out.append({
            "file": None,
            "line": None,
            "severity": m.group("sev"),
            "message": (m.group("msg") or "").strip(),
            "source": "ptxas",
        })
    return out


def build_error_feedback(errors: list[CompileError] | None,
                         source_context: str | None,
                         kind: FailureKind, *,
                         returncode: int | None = None,
                         max_diff: float | None = None,
                         out_bin: str | None = None,
                         timeout: int | float | None = None,
                         stdout_tail: str = "",
                         stderr_tail: str = "") -> str :
    """
    按失败类别拼装自我纠错反馈（纯函数，供闭环把报错喂回模型）。

    - errors：extract_compile_errors 的结构化列表（可为空）。
    - source_context：出错行附近的 .cu 源码片段（可 None，由上层读文件传入）。
    - kind：timeout / run_failed / no_output / wrong_result。
    有结构化编译错误时，优先用「行号 + 报错原因 + 出错行源码」；
    否则回落到 stdout/stderr 尾部（兜底，避免空反馈）。
    """
    def _compile_block() -> str :
        if not errors :
            return ""
        lines = ["编译器报错如下："]
        for e in errors[:5] :  # 只喂前 5 条，避免超长稀释关键信息
            loc = e.get("file") or "你的 .cu 文件"
            if e.get("line") is not None :
                loc = f"{loc}:{e['line']}"
            lines.append(f"  - {loc} [{e['severity']}] {e['message']}")
        block = "\n".join(lines)
        if source_context :
            block += "\n\n出错行附近的源码：\n" + source_context
        return block

    if kind == "timeout" :
        return (
            f"你上次的 CUDA 代码在 {timeout} 秒内未跑完（超时被强杀）。\n"
            f"[运行输出尾部]\n{stdout_tail}\n"
            f"[错误输出尾部]\n{stderr_tail}\n"
            f"可能是算法太慢（如 naive 三重循环）或存在死循环，请优化 kernel 后重新提供完整代码。"
        )

    if kind == "no_output" :
        return (
            f"你上次写的代码能编译运行，但没有输出结果矩阵 {out_bin}。\n"
            f"[stdout 尾部]\n{stdout_tail}\n"
            f"[stderr 尾部]\n{stderr_tail}\n"
            f"请确认 fwrite 写入了 {out_bin}，修复后重新提供完整的 CUDA 代码。"
        )

    if kind == "wrong_result" :
        # P1（3.3）：NaN 输出被正确判失败时，max_diff 可能是 NaN，直接格式化会出「nan」。
        md = sanitize_float(max_diff)
        diff_txt = f"{md:.6g}" if md is not None else "非有限值（NaN/Inf，结果矩阵含非法数值）"
        return (
            f"你上次写的代码能运行，但结果矩阵与基准不符（最大绝对误差 {diff_txt}，"
            f"超过校准容差）。\n请检查矩阵乘法逻辑（索引、累加、边界），修复后重新提供完整的 CUDA 代码。"
        )

    # run_failed：编译/运行失败，优先结构化编译错误，否则兜底尾部
    compile_block = _compile_block()
    if compile_block :
        return (
            f"你的代码在 Docker 沙箱中编译/运行失败（退出码 {returncode}）。\n"
            f"{compile_block}\n"
            f"请根据上面的行号和报错信息定位并修复，重新提供完整的 CUDA 代码。"
        )
    return (
        f"你的代码在 Docker 沙箱中运行失败（退出码 {returncode}）。\n"
        f"[stdout 尾部]\n{stdout_tail}\n"
        f"[stderr 尾部]\n{stderr_tail}\n"
        f"请根据报错信息定位问题，修复后重新提供完整的 CUDA 代码。"
    )


def build_result_payload(run_id: str,
                         timestamp: str,
                         matrix_size: int,
                         model_name: str,
                         prompt_version: str,
                         best: dict[str, Any],
                         records: list[dict[str, Any]],
                         env_snapshot: dict[str, Any],
                         knowledge: dict[str, Any] | None = None) -> dict[str, Any] :
    """
    构造 results schema v2 的落盘 payload（纯函数，不做 I/O）。

    schema v2 相比 v1 新增：run_id / timestamp / model 真实名（response.model，逐轮在
    attempts 里）/ prompt 版本 / 环境快照 / 每轮 code 全文与错误反馈，支撑
    「100 组数据」的论文复现与「瓶颈→策略→收益」证据链回溯。
    """
    def clean_best(b: dict[str, Any]) -> dict[str, Any] :
        return {
            "attempt": b.get("attempt"),
            "cu_filename": b.get("cu_filename"),
            "binary": b.get("binary"),
            "effective_time": sanitize_float(b.get("effective_time")),
            "self_reported_time": sanitize_float(b.get("self_reported_time")),
            "ncu_time_sec": sanitize_float(b.get("ncu_time_sec")),
            "max_diff": sanitize_float(b.get("max_diff")),
            "bottleneck": b.get("bottleneck"),
            "profile_source": b.get("profile_source"),
            "metrics": b.get("metrics"),
            "feedback": b.get("feedback"),
        }

    def clean_record(r: dict[str, Any]) -> dict[str, Any] :
        return {
            **r,
            "max_diff": sanitize_float(r.get("max_diff")),
            "time_sec": sanitize_float(r.get("time_sec")),
            "ncu_time_sec": sanitize_float(r.get("ncu_time_sec")),
        }

    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "matrix_size": matrix_size,
        "model": model_name,          # 配置的 API 别名（如 deepseek-chat）
        "prompt_version": prompt_version,
        "env": env_snapshot,
        "success": bool(best.get("ok")),
        "knowledge": knowledge,
        "best": clean_best(best),
        "attempts": [clean_record(r) for r in records],
    }


def write_result_json(payload: dict[str, Any], results_dir: str) -> str :
    """
    写入 results/{run_id}/{N}.json，返回路径。
    """
    run_dir = os.path.join(results_dir, str(payload["run_id"]))
    os.makedirs(run_dir, exist_ok = True)
    path = os.path.join(run_dir, f"{payload['matrix_size']}.json")
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(payload, f, ensure_ascii = False, indent = 2, default = str)
    return path


def append_run_index(payload: dict[str, Any], results_dir: str) -> str :
    """
    向 results/index.jsonl 追加一行 run 汇总，返回索引路径。
    """
    index_path = os.path.join(results_dir, "index.jsonl")
    summary = {
        "run_id": payload["run_id"],
        "timestamp": payload["timestamp"],
        "matrix_size": payload["matrix_size"],
        "model": payload["model"],
        "success": payload["success"],
        "best_attempt": payload["best"].get("attempt"),
        "best_effective_time": sanitize_float(payload["best"].get("effective_time")),
    }
    with open(index_path, "a", encoding = "utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii = False, default = str) + "\n")
    return index_path


def persist_result(payload: dict[str, Any], results_dir: str) -> tuple[str, str]:
    """
    写入 results/{run_id}/{N}.json 并追加 results/index.jsonl，返回 (json_path, index_path)。
    """
    path = write_result_json(payload, results_dir)
    index_path = append_run_index(payload, results_dir)
    return path, index_path
