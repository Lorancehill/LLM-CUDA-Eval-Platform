"""dashboard.py 的离线单测：results/ 聚合、NaN 净化、损坏索引行容错。"""

import json
import os
import shutil
import tempfile

import dashboard


def _write_run(root, run_id, size, best_ncu, ncu_vals, extra_index_line=None):
    """构造一个最小 schema v2 落盘（run 目录 + index 行）。"""
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    attempts = []
    for i, ncu in enumerate(ncu_vals, start=1):
        attempts.append({
            "attempt": i,
            "ok": ncu is not None,
            "time_sec": 0.001,
            "ncu_time_sec": ncu,
            "profile_source": "ncu" if ncu is not None else None,
            "bottleneck": "compute",
            "metrics": {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 90.0,
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 2.0,
                "sm__warps_active.avg.pct_of_peak_sustained_active": 90.0,
                "lts__t_sector_hit_rate.pct": 90.0,
            },
            "model": "deepseek-flash",
            "usage": {"total_tokens": 100},
        })
    payload = {
        "run_id": run_id,
        "timestamp": "2026-11-18T00:00:00+00:00",
        "matrix_size": size,
        "model": "deepseek-chat",
        "prompt_version": "matmul_v1",
        "env": {"gpu": "RTX 5060"},
        "success": best_ncu is not None,
        "knowledge": {"enabled": False},
        "best": {"attempt": len(attempts), "ncu_time_sec": best_ncu, "bottleneck": "compute"},
        "attempts": attempts,
    }
    with open(os.path.join(run_dir, f"{size}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)

    with open(os.path.join(root, "index.jsonl"), "a", encoding="utf-8") as f:
        if extra_index_line:
            f.write(extra_index_line + "\n")
        f.write(json.dumps({
            "run_id": run_id, "timestamp": payload["timestamp"],
            "matrix_size": size, "model": "deepseek-chat",
            "success": payload["success"], "best_attempt": len(attempts),
            "best_effective_time": best_ncu,
        }) + "\n")


def test_collect_results_empty():
    tmp = tempfile.mkdtemp()
    try:
        d = dashboard.collect_results(tmp)
        assert d["runs"] == []
        assert d["env"] is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_aggregates_and_sorts():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-b", 512, 1e-4, [2e-4, 1e-4])
        _write_run(tmp, "run-a", 256, 1e-5, [2e-5, 1e-5])
        d = dashboard.collect_results(tmp)
        # 按时间戳升序（同时间戳则稳定按 run_id 排序）
        assert [r["run_id"] for r in d["runs"]] == ["run-a", "run-b"]
        sizes = d["runs"][1]["sizes"]
        assert [s["matrix_size"] for s in sizes] == [512]
        s = sizes[0]
        assert s["success"] is True
        assert s["best_time_sec"] == 1e-4
        assert [a["ncu_time_sec"] for a in s["attempts"]] == [2e-4, 1e-4]
        a = s["attempts"][0]
        assert a["sm"] == 90.0 and a["dram"] == 2.0 and a["occ"] == 90.0 and a["l2"] == 90.0
        assert d["env"] == {"gpu": "RTX 5060"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_sanitizes_nan():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-nan", 256, None, [None])
        path = os.path.join(tmp, "run-nan", "256.json")
        data = json.loads(open(path, encoding="utf-8").read())
        data["attempts"][0]["metrics"]["sm__throughput.avg.pct_of_peak_sustained_elapsed"] = float("nan")
        data["best"]["ncu_time_sec"] = float("nan")
        data["best"]["effective_time"] = None
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, allow_nan=True, default=str)

        d = dashboard.collect_results(tmp)
        s = d["runs"][0]["sizes"][0]
        assert s["best_time_sec"] is None
        assert s["attempts"][0]["sm"] is None
        assert s["attempts"][0]["dram"] == 2.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_skips_corrupt_index_line():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-x", 256, 1e-5, [1e-5], extra_index_line="{not json")
        d = dashboard.collect_results(tmp)
        assert len(d["runs"]) == 1
        assert d["runs"][0]["run_id"] == "run-x"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_scans_run_dirs_without_index():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-scan", 256, 1e-5, [1e-5])
        os.remove(os.path.join(tmp, "index.jsonl"))
        d = dashboard.collect_results(tmp)
        assert len(d["runs"]) == 1
        assert d["runs"][0]["run_id"] == "run-scan"
        s = d["runs"][0]["sizes"][0]
        assert s["matrix_size"] == 256
        assert s["best_time_sec"] == 1e-5
        assert s["timing_source"] == "ncu"
        assert d["env"] == {"gpu": "RTX 5060"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_timing_source_and_knowledge():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-self", 512, None, [None])
        path = os.path.join(tmp, "run-self", "512.json")
        data = json.loads(open(path, encoding="utf-8").read())
        data["best"]["ncu_time_sec"] = None
        data["best"]["effective_time"] = 7.5e-5
        data["knowledge"] = {
            "enabled": True,
            "mode": "cards",
            "top_k": 3,
            "cards_loaded": 5,
            "retrieved_card_ids": ["c1", "c2"],
            "candidates_created": ["c3"],
            "knowledge_updates_applied": {"c1": {"current_confidence": 0.5, "suggested_confidence": 0.6}},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)

        d = dashboard.collect_results(tmp)
        s = d["runs"][0]["sizes"][0]
        assert s["timing_source"] == "self"
        assert s["best_time_sec"] == 7.5e-5
        assert s["knowledge_enabled"] is True
        assert s["knowledge"]["cards_loaded"] == 5
        assert s["knowledge"]["retrieved_card_ids"] == ["c1", "c2"]
        assert s["knowledge"]["candidates_created"] == ["c3"]
        assert s["knowledge"]["knowledge_updates_applied"]["c1"]["suggested_confidence"] == 0.6
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_collect_results_run_level_success():
    tmp = tempfile.mkdtemp()
    try:
        _write_run(tmp, "run-ok", 256, 1e-5, [1e-5])
        # 同 run 两个尺寸：256 成功、512 失败 → run 级 success 应为 False
        _write_run(tmp, "run-partial", 256, 1e-5, [1e-5])
        _write_run(tmp, "run-partial", 512, None, [None])
        d = dashboard.collect_results(tmp)
        by_id = {r["run_id"]: r for r in d["runs"]}
        assert by_id["run-ok"]["success"] is True
        assert by_id["run-partial"]["success"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
