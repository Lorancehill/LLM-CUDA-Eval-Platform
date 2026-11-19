#!/usr/bin/env python3
"""一键跑批脚本（P0-3）：参数化地以独立子进程批量运行 async_eval.py。

用法：
    python run_experiment.py --exp exp1 --sizes 256,512,1024 --kb 0,1 --repeats 3
    python run_experiment.py --exp exp1 --sizes 512 --kb 1 --repeats 1 --model deepseek-flash
    python run_experiment.py --exp exp1 --kb 0,1 --repeats 3 --dry-run

每个 run = 一个独立子进程跑 async_eval.py（干净状态），通过环境变量注入实验元数据：
    EXPERIMENT_ID / CONDITION / REPEAT / RUN_ID / MODEL / KNOWLEDGE_ENABLED / KNOWLEDGE_TOPK
- 子进程 stdout+stderr 落 experiments/{exp_id}/logs/{run_id}.log；
- async_eval 先把结果写进 results/{run_id}/，本脚本再归档（复制）到
  experiments/{exp_id}/runs/{run_id}/；
- 失败（退出码非 0）重试 1 次，连续失败 2 次标记 skipped 并记录原因，不中断整批；
- 结束后打印汇总表并写 experiments/{exp_id}/summary.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ASYNC_EVAL = os.path.join(HERE, "async_eval.py")
RESULTS_DIR = os.path.join(HERE, "results")

_KB_ON = {"1", "true", "on", "yes"}


def _kb_enabled(kb: str) -> bool:
    """kb 取值 → KNOWLEDGE_ENABLED（1=开，0=关）。"""
    return kb.strip().lower() in _KB_ON


def build_cells(args) -> list[dict]:
    """展开 (kb × repeat) 跑批矩阵。每个 cell 对应一个独立子进程。"""
    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    kbs = [x for x in args.kb.split(",") if x.strip()]
    cells = []
    for kb in kbs:
        cond = f"{args.cond}{kb}"
        for repeat in range(1, args.repeats + 1):
            cells.append({
                "kb": kb,
                "condition": cond,
                "repeat": repeat,
                "sizes": sizes,
                "knowledge_enabled": _kb_enabled(kb),
            })
    return cells


def _run_id(exp: str, cond: str, repeat: int) -> str:
    """确定性前缀 + 随机后缀，保证重试不撞目录、仍可肉眼归组。"""
    return f"{exp}-{cond}-r{repeat}-{uuid.uuid4().hex[:6]}"


def _child_env(args, cell: dict, run_id: str) -> dict:
    env = os.environ.copy()
    env["EXPERIMENT_ID"] = args.exp
    env["CONDITION"] = cell["condition"]
    env["REPEAT"] = str(cell["repeat"])
    env["RUN_ID"] = run_id
    env["KNOWLEDGE_ENABLED"] = "1" if cell["knowledge_enabled"] else "0"
    if args.topk is not None:
        env["KNOWLEDGE_TOPK"] = str(args.topk)
    if args.model:
        env["MODEL"] = args.model
    # 子进程 stdout 是管道，默认块缓冲会让日志迟迟不落盘；强制行缓冲
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_one(args, cell: dict, run_id: str) -> tuple[int, str]:
    """跑一个子进程，返回 (returncode, 错误尾部)。stdout/stderr 落日志文件。"""
    log_dir = os.path.join(HERE, "experiments", args.exp, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")
    cmd = [args.python, ASYNC_EVAL]
    err_tail = ""
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                cmd, cwd=HERE, env=_child_env(args, cell, run_id),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=args.run_timeout,
            )
            return proc.returncode, ""
        except subprocess.TimeoutExpired:
            err_tail = f"run 超时（>{args.run_timeout}s）"
            log.write(f"\n[run_experiment] {err_tail}\n")
            return 1, err_tail
        except Exception as e:
            err_tail = f"子进程启动失败：{e}"
            log.write(f"\n[run_experiment] {err_tail}\n")
            return 1, err_tail


def read_run_metrics(run_id: str, sizes: list[int]) -> dict:
    """从 results/{run_id}/{size}.json 读每个尺寸的 success + best effective_time。"""
    metrics = {"sizes": {}}
    for size in sizes:
        path = os.path.join(RESULTS_DIR, run_id, f"{size}.json")
        entry = {"success": None, "best_sec": None}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                entry["success"] = bool(data.get("success"))
                best = data.get("best") or {}
                entry["best_sec"] = best.get("effective_time")
                # 环境快照（代码版本等）只读一次即可（同一 run 内各尺寸一致）
                env = data.get("env") or {}
                for k in ("git_commit", "git_dirty", "EXPERIMENT_ID", "CONDITION", "REPEAT"):
                    if k not in metrics:
                        metrics[k] = env.get(k)
            except (json.JSONDecodeError, OSError):
                pass
        metrics["sizes"][size] = entry
    return metrics


def archive_run(args, run_id: str) -> str | None:
    """把 results/{run_id}/ 复制到 experiments/{exp}/runs/{run_id}/，返回目标路径。"""
    src = os.path.join(RESULTS_DIR, run_id)
    if not os.path.isdir(src):
        return None
    dst = os.path.join(HERE, "experiments", args.exp, "runs", run_id)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def _fmt_sec(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return "—"


def _summary_row(args, cell: dict, run_id: str, status: str, metrics: dict,
                 model: str | None, note: str) -> dict:
    row = {
        "exp_id": args.exp,
        "run_id": run_id,
        "condition": cell["condition"],
        "kb": cell["kb"],
        "repeat": cell["repeat"],
        "model": model or "",
        "status": status,
        "n_ok": sum(1 for e in metrics["sizes"].values() if e["success"]),
        "n_sizes": len(metrics["sizes"]),
        "git_commit": metrics.get("git_commit") or "",
        "git_dirty": metrics.get("git_dirty"),
    }
    for size, e in metrics["sizes"].items():
        row[f"best_{size}_sec"] = e["best_sec"]
    if note:
        row["note"] = note
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="一键跑批 async_eval.py（P0-3）")
    ap.add_argument("--exp", default="exp1", help="实验 ID（也是 experiments/{exp} 目录名，EXPERIMENT_ID）")
    ap.add_argument("--sizes", default="256,512,1024", help="矩阵尺寸，逗号分隔")
    ap.add_argument("--kb", default="0,1", help="知识库消融取值，逗号分隔（1=开 0=关）")
    ap.add_argument("--repeats", type=int, default=3, help="每个条件的重复次数")
    ap.add_argument("--model", default=None, help="MODEL 环境变量（缺省用 async_eval 默认 deepseek-chat）")
    ap.add_argument("--cond", default="kb", help="条件标签前缀（默认 kb，条件名 = 前缀+kb 值，如 kb0/kb1）")
    ap.add_argument("--topk", type=int, default=None, help="KNOWLEDGE_TOPK 覆盖（缺省用 async_eval 默认 3）")
    ap.add_argument("--run-timeout", type=int, default=1800, help="单个 run 的超时秒数（默认 1800）")
    ap.add_argument("--python", default=sys.executable, help="运行 async_eval.py 的解释器（默认当前解释器）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不真正执行")
    args = ap.parse_args()

    cells = build_cells(args)
    print(f"实验 {args.exp}：{len(cells)} 个 run = "
          f"{len(set(c['kb'] for c in cells))} kb 条件 × {args.repeats} 重复，"
          f"尺寸 {args.sizes}，模型 {args.model or 'deepseek-chat（默认）'}")

    if args.dry_run:
        for i, c in enumerate(cells, 1):
            print(f"  [{i:2d}] condition={c['condition']} repeat={c['repeat']} "
                  f"kb={c['kb']} KNOWLEDGE_ENABLED={int(c['knowledge_enabled'])}")
        print("（--dry-run：未执行任何子进程）")
        return 0

    rows = []
    for i, cell in enumerate(cells, 1):
        status = "ok"
        final_run_id = None
        final_metrics = None
        note = ""
        for attempt in (1, 2):
            run_id = _run_id(args.exp, cell["condition"], cell["repeat"])
            print(f"[{i}/{len(cells)}] {cell['condition']} repeat={cell['repeat']} "
                  f"attempt={attempt} run_id={run_id} ...", flush=True)
            rc, err_tail = run_one(args, cell, run_id)
            archive_run(args, run_id)
            metrics = read_run_metrics(run_id, cell["sizes"])
            if rc == 0:
                final_run_id, final_metrics, status, note = run_id, metrics, "ok", ""
                print(f"    ✅ 完成（{sum(1 for e in metrics['sizes'].values() if e['success'])}/"
                      f"{len(metrics['sizes'])} 尺寸成功）", flush=True)
                break
            if attempt == 1:
                print(f"    ⚠️ 失败（退出码 {rc}），重试 1 次...", flush=True)
                continue
            final_run_id, final_metrics, status = run_id, metrics, "skipped"
            note = err_tail or f"退出码 {rc}"
            print(f"    ❌ 连续失败 2 次，标记 skipped（{note}）", flush=True)

        rows.append(_summary_row(args, cell, final_run_id, status, final_metrics,
                                 args.model, note))

    # 汇总表（控制台）
    print("\n========== 汇总 ==========")
    hdr = ["condition", "repeat", "status", "n_ok", "best(s)"]
    for r in rows:
        bests = " | ".join(_fmt_sec(r.get(f"best_{s}_sec")) for s in cells[0]["sizes"])
        print(f"  {r['condition']:<8} r{r['repeat']}  {r['status']:<8} "
              f"{r['n_ok']}/{r['n_sizes']}  {bests}")

    # summary.csv（动态列：每个尺寸一列 best）
    exp_dir = os.path.join(HERE, "experiments", args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    csv_path = os.path.join(exp_dir, "summary.csv")
    size_cols = [f"best_{s}_sec" for s in cells[0]["sizes"]]
    fieldnames = ["exp_id", "run_id", "condition", "kb", "repeat", "model", "status",
                  "n_ok", "n_sizes", "git_commit", "git_dirty"] + size_cols + ["note"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n汇总已写：{csv_path}")
    print(f"结果归档：{os.path.join(exp_dir, 'runs')}")
    print(f"日志归档：{os.path.join(exp_dir, 'logs')}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
