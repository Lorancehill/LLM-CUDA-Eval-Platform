"""评测纯逻辑：耗时解析、best-so-far 判据、结果落盘（results schema v2）。

从 async_eval 抽出，便于在**不导入 async_eval**（它 import 即创建 API client）的前提下
单测这些逻辑。函数不依赖 GPU / Docker / API。
"""

from __future__ import annotations

import json
import os
import re


_TIME_RE = re.compile(r"Time:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")


def parse_time_seconds(stdout_str: str) -> float | None:
    """从 CUDA 输出解析 'Time: xxx seconds' 中的秒数；无匹配返回 None。"""
    m = _TIME_RE.search(stdout_str or "")
    return float(m.group(1)) if m else None


def parse_time_seconds_median(stdout_str: str) -> float | None:
    """解析全部 'Time: xxx seconds' 并返回中位数（论文计时口径：多次重复取中位数，
    抗首轮冷启动/偶发抖动）；无匹配返回 None。"""
    vals = [float(m.group(1)) for m in _TIME_RE.finditer(stdout_str or "")]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def should_update_best(best_ok: bool, best_time: float | None, new_time: float | None) -> bool:
    """best-so-far 更新判据（纯函数）。

    - 尚无正确结果（best_ok=False）：无条件记录（首个正确结果）。
    - 已有正确结果：
        - 旧耗时缺失、新耗时存在 → 更新（能测得就更可信，补齐缺失计时）；
        - 新旧都测得且新耗时严格更小 → 更新；
        - 否则（无法证明更快）保留旧 best，避免被改坏的内核「反杀」。
    """
    if not best_ok:
        return True
    if best_time is None:
        return new_time is not None
    return new_time is not None and new_time < best_time


def sanitize_float(v: float | None) -> float | None:
    """NaN -> None（json.dump 默认会写非法 NaN），其余原样。"""
    return None if (v is not None and v != v) else v


def compute_speedup(prev_time, cur_time) -> float | None:
    """加速比 = prev_time / cur_time（>1 更快，<1 更慢，==1 持平）。

    两者都必须是正的有限数才可比较；否则返回 None（无法比较）。
    用于记录「本轮相对上一成功轮的提速」，与 best-so-far 的绝对保优判据互补。
    """
    try:
        p = float(prev_time)
        c = float(cur_time)
    except (TypeError, ValueError):
        return None
    if p != p or c != c:   # NaN
        return None
    if p <= 0 or c <= 0:
        return None
    return p / c


def build_result_payload(run_id, timestamp, matrix_size, model_name, prompt_version,
                         best, records, env_snapshot, knowledge=None):
    """构造 results schema v2 的落盘 payload（纯函数，不做 I/O）。

    schema v2 相比 v1 新增：run_id / timestamp / model 真实名（response.model，逐轮在
    attempts 里）/ prompt 版本 / 环境快照 / 每轮 code 全文与错误反馈，支撑
    「100 组数据」的论文复现与「瓶颈→策略→收益」证据链回溯。
    """
    def clean_best(b):
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

    def clean_record(r):
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


def write_result_json(payload: dict, results_dir: str) -> str:
    """写入 results/{run_id}/{N}.json，返回路径。"""
    run_dir = os.path.join(results_dir, str(payload["run_id"]))
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{payload['matrix_size']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def append_run_index(payload: dict, results_dir: str) -> str:
    """向 results/index.jsonl 追加一行 run 汇总，返回索引路径。"""
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
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, default=str) + "\n")
    return index_path


def persist_result(payload: dict, results_dir: str) -> tuple[str, str]:
    """写入 results/{run_id}/{N}.json 并追加 results/index.jsonl，返回 (json_path, index_path)。"""
    path = write_result_json(payload, results_dir)
    index_path = append_run_index(payload, results_dir)
    return path, index_path
