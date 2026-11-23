"""
评测纯逻辑：耗时解析、best-so-far 判据、结果落盘（results schema v2）。

从 async_eval 抽出，便于在**不导入 async_eval**（它 import 即创建 API client）的前提下
单测这些逻辑。函数不依赖 GPU / Docker / API。

健壮性约定（v2.1）：解析入口统一容忍 None / bytes / 畸形值，不因脏输入崩溃；数值判据
统一按「有限浮点」归一化（NaN / ±Inf / 超大整数都视为不可比较）；payload 落盘前做深层
JSON 净化（任意层级 NaN/Inf → null）；results/{N}.json 用「临时文件 + os.replace」原子
替换，results/index.jsonl 用 O_APPEND 单次整行追加，避免中断留下半截文件、降低并发交错。
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from typing import Any, Literal, TypedDict

SCHEMA_VERSION = 2


class CompileError(TypedDict) :
    """
    一条结构化编译错误（extract_compile_errors 的返回元素）。
    """

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


def _coerce_text(value: object) -> str :
    """
    把任意输入统一成 str（解析入口的防御层）。

    - None → ""（大量调用方会传 None，等价于空输出）；
    - bytes/bytearray → UTF-8 解码（非法字节用 replacement 字符兜底，不抛异常）；
    - 其它类型 → str(value)（例如 int/异常对象，便于日志诊断）。
    """
    if value is None :
        return ""
    if isinstance(value, (bytes, bytearray)) :
        return bytes(value).decode("utf-8", errors = "replace")
    return value if isinstance(value, str) else str(value)


def _to_finite_float(value: object) -> float | None :
    """
    转成有限 float；None / 不可转换 / 溢出 / NaN / ±Inf 一律返回 None。

    与 sanitize_float 的区别：本函数只服务「数值判据」（比较/除法），非法值统一视为
    「不可比较」；sanitize_float 用于落盘前的字段清洗，对非数值输入保留原样。
    """
    if value is None :
        return None
    try :
        f = float(value)
    except (TypeError, ValueError, OverflowError) :
        return None
    return f if math.isfinite(f) else None


def _json_safe(value: Any) -> Any :
    """
    深拷贝式 JSON 净化：任意层级的 NaN/Inf → None，tuple → list。

    不修改原对象（payload 仍然可被调用方继续使用）；dict 键保持原样交给 json 处理。
    """
    if isinstance(value, dict) :
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)) :
        return [_json_safe(v) for v in value]
    if isinstance(value, float) :
        return value if math.isfinite(value) else None
    return value


def parse_time_seconds(stdout_str: str | bytes | None) -> float | None :
    """
    从 CUDA 输出解析 'Time: xxx seconds' 中的秒数；无匹配返回 None。
    """
    m = _TIME_RE.search(_coerce_text(stdout_str))

    return float(m.group(1)) if m else None


def parse_time_seconds_median(stdout_str: str | bytes | None) -> float | None :
    """
    解析全部 'Time: xxx seconds' 并返回中位数（论文计时口径：多次重复取中位数，
    抗首轮冷启动/偶发抖动）；无匹配返回 None。
    """
    vals = [float(m.group(1)) for m in _TIME_RE.finditer(_coerce_text(stdout_str))]
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
    best_t = _to_finite_float(best_time)
    new_t = _to_finite_float(new_time)

    if not best_ok :
        return True
    if best_t is None :
        return new_t is not None
    return new_t is not None and new_t < best_t


def sanitize_float(v: float | None) -> float | None :
    """
    非有限值（NaN/Inf）-> None，其余原样。

    json.dump 默认会写出非法 JSON 的 NaN/Infinity；落盘前统一清洗。
    无法转成 float 的输入原样返回（调用方语义不变）；转换溢出（如超大 int）返回 None。
    """
    if v is None :
        return None
    try :
        f = float(v)
    except OverflowError :
        return None
    except (TypeError, ValueError) :
        return v
    return None if not math.isfinite(f) else v


def compute_speedup(prev_time: float | None, cur_time: float | None) -> float | None :
    """
    加速比 = prev_time / cur_time（>1 更快，<1 更慢，==1 持平）。

    两者都必须是正的有限数才可比较；否则返回 None（无法比较）。
    用于记录「本轮相对上一成功轮的提速」，与 best-so-far 的绝对保优判据互补。
    """
    p = _to_finite_float(prev_time)
    c = _to_finite_float(cur_time)
    if p is None or c is None :
        return None
    if p <= 0 or c <= 0 :
        return None
    return p / c


def extract_compile_errors(text: str | bytes | None) -> list[CompileError] :
    """
    从 nvcc / g++ / ptxas 编译输出里提取结构化报错行（file/line/severity/message）。

    自我纠错闭环用：把「具体行号 + 报错原因」喂回模型，而不是只给 stderr 尾部 300 字符。
    覆盖三种形态（示例见 _NVCC_ERR_RE / _GCC_ERR_RE / _PTXAS_ERR_RE）：
    - nvcc："matrix_mul_256.cu(12): error: identifier "foo" is undefined"
    - 宿主 g++："/tmp/.../main.cu:12:5: error: 'foo' was not declared"
    - ptxas（编译期、无行号）："ptxas error : Entry function ... uses too much shared data"
    无匹配返回空列表。
    """
    text = _coerce_text(text)
    out: list[CompileError] = []
    for m in _NVCC_ERR_RE.finditer(text) :
        out.append({
            "file": m.group("file") or None,
            "line": int(m.group("line")),
            "severity": m.group("sev"),
            "message": (m.group("msg") or "").strip(),
            "source": "nvcc",
        })
    for m in _GCC_ERR_RE.finditer(text) :
        out.append({
            "file": m.group("file") or None,
            "line": int(m.group("line")),
            "severity": m.group("sev"),
            "message": (m.group("msg") or "").strip(),
            "source": "gcc",
        })
    for m in _PTXAS_ERR_RE.finditer(text) :
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
                         timeout: float | None = None,
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
    # 防御：errors 可能是 None / 单个 dict / 混入非 dict 元素（上层解析异常时）
    if isinstance(errors, dict) :
        normalized_errors: list[dict[str, Any]] = [errors]
    elif isinstance(errors, (list, tuple)) :
        normalized_errors = [e for e in errors if isinstance(e, dict)]
    else :
        normalized_errors = []

    source_context = _coerce_text(source_context) or None
    stdout_tail = _coerce_text(stdout_tail)
    stderr_tail = _coerce_text(stderr_tail)
    out_bin_text = _coerce_text(out_bin) if out_bin is not None else "None"

    def _compile_block() -> str :
        if not normalized_errors :
            return ""
        lines = ["编译器报错如下："]
        for e in normalized_errors[:5] :  # 只喂前 5 条，避免超长稀释关键信息
            file = e.get("file") or "你的 .cu 文件"
            line = e.get("line")
            loc = f"{file}:{line}" if line is not None else file
            severity = e.get("severity") or ""
            message = e.get("message") or ""
            lines.append(f"  - {loc} [{severity}] {message}")
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
            f"你上次写的代码能编译运行，但没有输出结果矩阵 {out_bin_text}。\n"
            f"[stdout 尾部]\n{stdout_tail}\n"
            f"[stderr 尾部]\n{stderr_tail}\n"
            f"请确认 fwrite 写入了 {out_bin_text}，修复后重新提供完整的 CUDA 代码。"
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
    # 防御：best/records/env 可能是 None 或类型不对（例如上层异常路径只传了部分字段）
    best = {} if best is None else best
    records = [] if records is None else records
    env_snapshot = {} if env_snapshot is None else env_snapshot

    if not isinstance(best, dict) :
        raise TypeError(f"best 必须是 dict，实际是 {type(best).__name__}")
    if not isinstance(records, (list, tuple)) :
        raise TypeError(f"records 必须是 list，实际是 {type(records).__name__}")
    if not isinstance(env_snapshot, dict) :
        raise TypeError(f"env_snapshot 必须是 dict，实际是 {type(env_snapshot).__name__}")

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

    def clean_record(record: dict[str, Any]) -> dict[str, Any] :
        return {
            **record,
            "max_diff": sanitize_float(record.get("max_diff")),
            "time_sec": sanitize_float(record.get("time_sec")),
            "ncu_time_sec": sanitize_float(record.get("ncu_time_sec")),
        }

    attempts = []
    for index, record in enumerate(records) :
        if not isinstance(record, dict) :
            raise TypeError(f"records[{index}] 必须是 dict，实际是 {type(record).__name__}")
        attempts.append(clean_record(record))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp": timestamp,
        "matrix_size": matrix_size,
        "model": model_name,          # 配置的 API 别名（如 deepseek-chat）
        "prompt_version": prompt_version,
        "env": env_snapshot,
        "success": bool(best.get("ok")),
        "knowledge": knowledge,
        "best": clean_best(best),
        "attempts": attempts,
    }
    # 构造即净化：任意层级 NaN/Inf -> None、tuple -> list，调用方拿到的 payload 可直接 json
    return _json_safe(payload)


def write_result_json(payload: dict[str, Any], results_dir: str) -> str :
    """
    写入 results/{run_id}/{N}.json，返回路径。

    采用「同目录临时文件 + os.replace」原子替换：进程中断/磁盘写满时不会留下半截 JSON。
    """
    if not isinstance(payload, dict) :
        raise TypeError(f"payload 必须是 dict，实际是 {type(payload).__name__}")
    try :
        run_id = payload["run_id"]
        matrix_size = payload["matrix_size"]
    except KeyError as exc :
        raise ValueError(f"payload 缺少必需字段：{exc}") from exc

    run_dir = os.path.join(results_dir, str(run_id))
    os.makedirs(run_dir, exist_ok = True)
    path = os.path.join(run_dir, f"{matrix_size}.json")

    fd, tmp_path = tempfile.mkstemp(prefix = f".{matrix_size}.", suffix = ".json.tmp", dir = run_dir)
    try :
        with os.fdopen(fd, "w", encoding = "utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii = False, indent = 2,
                      default = str, allow_nan = False)
        os.chmod(tmp_path, 0o644)   # mkstemp 默认 0600；对齐普通 open() 的 0644
        os.replace(tmp_path, path)
    except BaseException :
        try :
            os.unlink(tmp_path)
        except OSError :
            pass
        raise
    return path


def append_run_index(payload: dict[str, Any], results_dir: str) -> str :
    """
    向 results/index.jsonl 追加一行 run 汇总，返回索引路径。

    用 O_APPEND + 单次 write（循环写完）追加整行，降低多尺寸并发写 index 的交错风险；
    在旧字段基础上新增 schema_version，供看板/分析脚本做版本兼容（旧行无该字段仍可读）。
    """
    if not isinstance(payload, dict) :
        raise TypeError(f"payload 必须是 dict，实际是 {type(payload).__name__}")

    best = payload.get("best") or {}
    if not isinstance(best, dict) :
        best = {}

    index_path = os.path.join(results_dir, "index.jsonl")
    os.makedirs(results_dir or ".", exist_ok = True)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": payload.get("run_id"),
        "timestamp": payload.get("timestamp"),
        "matrix_size": payload.get("matrix_size"),
        "model": payload.get("model"),
        "success": payload.get("success"),
        "best_attempt": best.get("attempt"),
        "best_effective_time": sanitize_float(best.get("effective_time")),
    }
    line = json.dumps(summary, ensure_ascii = False, default = str,
                      allow_nan = False) + "\n"

    fd = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    try :
        data = line.encode("utf-8")
        while data :   # os.write 可能短写；循环直到整行写完
            written = os.write(fd, data)
            data = data[written:]
    finally :
        os.close(fd)
    return index_path


def persist_result(payload: dict[str, Any], results_dir: str) -> tuple[str, str] :
    """
    写入 results/{run_id}/{N}.json 并追加 results/index.jsonl，返回 (json_path, index_path)。
    """
    path = write_result_json(payload, results_dir)
    index_path = append_run_index(payload, results_dir)
    return path, index_path


__all__ = [
    "SCHEMA_VERSION",
    "CompileError",
    "FailureKind",
    "append_run_index",
    "build_error_feedback",
    "build_result_payload",
    "compute_speedup",
    "extract_compile_errors",
    "parse_time_seconds",
    "parse_time_seconds_median",
    "persist_result",
    "sanitize_float",
    "should_update_best",
    "write_result_json",
]
