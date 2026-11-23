"""eval_logic 健壮性回归测试（v2.1）：脏输入 / 极端值 / 原子落盘 / 并发，与 test_eval_logic.py 互补。"""

import json
import os
import shutil
import tempfile
import threading

import eval_logic


def _payload(**overrides):
    base = {
        "run_id": "run-rob",
        "timestamp": "2026-11-20T00:00:00+00:00",
        "matrix_size": 256,
        "model_name": "deepseek-chat",
        "prompt_version": "matmul_v1",
        "best": {"ok": True, "attempt": 1, "effective_time": 1e-5},
        "records": [],
        "env_snapshot": {"python": "3.14"},
    }
    base.update(overrides)
    return eval_logic.build_result_payload(**base)


def test_sanitize_float_overflow_returns_none():
    # 超大 int 无法转 float（OverflowError）→ 视为不可用，返回 None，不向上抛异常
    assert eval_logic.sanitize_float(10 ** 400) is None
    assert eval_logic.sanitize_float(float("nan")) is None
    assert eval_logic.sanitize_float("x") == "x"      # 非数值保持原样（旧语义）


def test_should_update_best_recovers_from_nan():
    # NaN 旧 best 不应把后续可测结果挡在门外
    assert eval_logic.should_update_best(True, float("nan"), 0.3) is True
    assert eval_logic.should_update_best(True, 0.3, float("nan")) is False
    assert eval_logic.should_update_best(True, None, None) is False


def test_compute_speedup_rejects_overflow_and_nan():
    assert eval_logic.compute_speedup(10 ** 400, 1.0) is None
    assert eval_logic.compute_speedup(1.0, float("nan")) is None
    assert eval_logic.compute_speedup(0.5, 0.1) == 5.0


def test_parse_time_accepts_bytes_and_none():
    assert eval_logic.parse_time_seconds(b"Time: 0.5 seconds") == 0.5
    assert eval_logic.parse_time_seconds(None) is None
    assert eval_logic.parse_time_seconds(123) is None
    assert eval_logic.parse_time_seconds_median(
        b"Time: 1 seconds\nTime: 3 seconds\nTime: 5 seconds") == 3.0


def test_extract_compile_errors_accepts_bytes():
    errors = eval_logic.extract_compile_errors(
        b'matrix_mul_256.cu(12): error: identifier "foo" is undefined')
    assert len(errors) == 1 and errors[0]["line"] == 12
    assert eval_logic.extract_compile_errors(None) == []


def test_build_error_feedback_tolerates_malformed_errors():
    # 缺字段 / 混入非 dict / errors 本身不是 list → 都不能崩
    fb = eval_logic.build_error_feedback(
        [{"file": None}, {"severity": "error"}, "junk", 3], None, "run_failed",
        returncode=1)
    assert "退出码 1" in fb
    assert "退出码" in eval_logic.build_error_feedback(123, None, "run_failed",
                                                     returncode=2)
    fb2 = eval_logic.build_error_feedback(
        [{"file": "a.cu", "line": 1, "severity": "error", "message": "x",
          "source": "nvcc"}],
        b">>> 1 | bad();", "run_failed")
    assert "出错行附近的源码" in fb2


def test_build_result_payload_deep_sanitizes_and_versions():
    best = {"ok": True, "attempt": 1, "effective_time": 1e-5,
            "max_diff": float("nan"),
            "metrics": {"gpu__time_duration.sum": float("nan"),
                        "nested": {"vals": [float("inf"), 1.0], "t": (1, 2)}}}
    payload = _payload(best=best)
    assert payload["schema_version"] == 2
    assert payload["best"]["max_diff"] is None
    assert payload["best"]["metrics"]["gpu__time_duration.sum"] is None
    assert payload["best"]["metrics"]["nested"]["vals"] == [None, 1.0]
    assert payload["best"]["metrics"]["nested"]["t"] == [1, 2]      # tuple → list


def test_write_result_json_atomic_and_strict():
    tmp = tempfile.mkdtemp()
    try:
        payload = _payload(best={"ok": True, "max_diff": float("nan")})
        path = eval_logic.write_result_json(payload, tmp)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        data = json.loads(raw)                       # 严格 JSON（无 NaN/Infinity）
        assert data["schema_version"] == 2
        assert "NaN" not in raw and "Infinity" not in raw
        leftovers = [n for n in os.listdir(os.path.dirname(path))
                     if n.endswith(".json.tmp")]
        assert leftovers == []                       # 不留临时文件
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_write_result_json_missing_keys_raises():
    tmp = tempfile.mkdtemp()
    try:
        try:
            eval_logic.write_result_json({"run_id": "x"}, tmp)
            assert False, "应当抛 ValueError"
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_append_run_index_concurrent_lines():
    # 多线程并发追加：行数不丢、每行都是合法 JSON（O_APPEND 单次写）
    tmp = tempfile.mkdtemp()
    try:
        def worker(i):
            eval_logic.append_run_index(_payload(run_id=f"r{i}"), tmp)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(os.path.join(tmp, "index.jsonl"), encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line]
        assert len(lines) == 10
        assert all(json.loads(l) for l in lines)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_persist_result_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        payload = _payload()
        path, index_path = eval_logic.persist_result(payload, tmp)
        assert os.path.exists(path) and os.path.exists(index_path)
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["run_id"] == "run-rob"
        with open(index_path, encoding="utf-8") as f:
            line = f.read().strip()
        summary = json.loads(line)
        assert summary["run_id"] == "run-rob"
        assert summary["schema_version"] == eval_logic.SCHEMA_VERSION
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
