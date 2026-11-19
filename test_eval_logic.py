"""eval_logic.py 的离线单测：耗时解析、best-so-far 判据、落盘 schema v2。"""

import json
import os
import shutil
import tempfile

import eval_logic


def test_parse_time_seconds():
    assert eval_logic.parse_time_seconds("Time: 0.123456 seconds") == 0.123456
    assert eval_logic.parse_time_seconds("Time: 1.5e-3 seconds") == 1.5e-3
    assert eval_logic.parse_time_seconds("no time here") is None


def test_parse_time_seconds_median():
    # 5 次重复取中位数（抗首轮冷启动/抖动）
    out = "\n".join(f"Time: {t} seconds" for t in [1.0, 2.0, 3.0, 4.0, 100.0])
    assert eval_logic.parse_time_seconds_median(out) == 3.0
    # 偶数个取中间两数均值
    out2 = "\n".join(f"Time: {t} seconds" for t in [1.0, 2.0, 3.0, 4.0])
    assert eval_logic.parse_time_seconds_median(out2) == 2.5
    # 单个值 = 它本身
    assert eval_logic.parse_time_seconds_median("Time: 0.5 seconds") == 0.5
    # 无匹配 → None
    assert eval_logic.parse_time_seconds_median("no time") is None


def test_should_update_best_boundary():
    # 首次正确：无条件记录
    assert eval_logic.should_update_best(False, None, None) is True
    assert eval_logic.should_update_best(False, None, 0.5) is True
    # 更快 → 更新
    assert eval_logic.should_update_best(True, 0.5, 0.3) is True
    # 更慢 → 保留
    assert eval_logic.should_update_best(True, 0.5, 0.7) is False
    # 相等 → 保留
    assert eval_logic.should_update_best(True, 0.5, 0.5) is False
    # 旧缺失、新存在 → 更新（B6 修复）
    assert eval_logic.should_update_best(True, None, 0.3) is True
    # 旧存在、新缺失 → 保留
    assert eval_logic.should_update_best(True, 0.5, None) is False
    # 两者缺失 → 保留
    assert eval_logic.should_update_best(True, None, None) is False


def test_sanitize_float():
    assert eval_logic.sanitize_float(float("nan")) is None
    assert eval_logic.sanitize_float(1.5) == 1.5
    assert eval_logic.sanitize_float(None) is None


def test_compute_speedup():
    assert eval_logic.compute_speedup(0.5, 0.1) == 5.0
    assert eval_logic.compute_speedup(0.5, 0.5) == 1.0
    assert eval_logic.compute_speedup(0.1, 0.5) == 0.2
    assert eval_logic.compute_speedup(None, 0.5) is None
    assert eval_logic.compute_speedup(0.5, None) is None
    assert eval_logic.compute_speedup(0.0, 0.5) is None
    assert eval_logic.compute_speedup(0.5, 0.0) is None
    assert eval_logic.compute_speedup(0.5, float("nan")) is None
    assert eval_logic.compute_speedup("x", 0.5) is None


def test_extract_compile_errors_nvcc():
    err = (
        "matrix_mul_256.cu(12): error: identifier \"foo\" is undefined\n"
        "          detected during instantiation of \"void matmul_kernel<float>(...)\"\n"
        "(24): here\n"
        "1 error detected in the compilation of \"matrix_mul_256.cu\".\n"
    )
    got = eval_logic.extract_compile_errors(err)
    assert len(got) == 1
    assert got[0]["file"] == "matrix_mul_256.cu"
    assert got[0]["line"] == 12
    assert got[0]["severity"] == "error"
    assert got[0]["source"] == "nvcc"
    assert "foo" in got[0]["message"]


def test_extract_compile_errors_gcc_and_ptxas():
    err = (
        "/tmp/tmpxft_000/main.cu:18:5: error: 'bar' was not declared in this scope\n"
        "ptxas error   : Entry function '_Z13matmul_kernelPfS_S_i' uses too much shared data\n"
    )
    got = eval_logic.extract_compile_errors(err)
    assert len(got) == 2
    assert got[0]["source"] == "gcc" and got[0]["line"] == 18 and got[0]["file"] == "/tmp/tmpxft_000/main.cu"
    assert got[1]["source"] == "ptxas" and got[1]["line"] is None and "shared" in got[1]["message"]


def test_extract_compile_errors_no_match():
    assert eval_logic.extract_compile_errors("") == []
    assert eval_logic.extract_compile_errors("Time: 0.5 seconds\nnormal output") == []


def test_build_error_feedback_prefers_compile_errors():
    errors = eval_logic.extract_compile_errors(
        "matrix_mul_256.cu(12): error: identifier \"foo\" is undefined"
    )
    ctx = ">>>   12 | int foo = x;\n      13 | "
    fb = eval_logic.build_error_feedback(errors, ctx, "run_failed", returncode=1,
                                         stdout_tail="tail-out", stderr_tail="tail-err")
    assert "matrix_mul_256.cu:12" in fb          # 具体行号喂回模型
    assert "foo" in fb
    assert "出错行附近的源码" in fb
    assert "tail-err" not in fb                  # 有结构化错误时不再兜底尾部


def test_build_error_feedback_fallback_to_tail():
    fb = eval_logic.build_error_feedback([], None, "run_failed", returncode=2,
                                         stdout_tail="out", stderr_tail="err")
    assert "err" in fb and "退出码 2" in fb


def test_build_error_feedback_wrong_result_and_timeout():
    fb = eval_logic.build_error_feedback([], None, "wrong_result", max_diff=1.5)
    assert "1.5" in fb and "容差" in fb
    fb2 = eval_logic.build_error_feedback([], None, "timeout", timeout=20)
    assert "20 秒" in fb2 and "超时" in fb2


def test_build_error_feedback_wrong_result_nan_sanitized():
    # P1（3.3）：NaN 输出被判失败时，反馈不应出现「最大绝对误差 nan」
    fb = eval_logic.build_error_feedback([], None, "wrong_result", max_diff=float("nan"))
    assert "最大绝对误差 nan" not in fb
    assert "非有限值" in fb


def test_persist_result_writes_json_and_index():
    tmp = tempfile.mkdtemp()
    try:
        payload = eval_logic.build_result_payload(
            run_id="run-1",
            timestamp="2026-11-18T00:00:00Z",
            matrix_size=256,
            model_name="deepseek-chat",
            prompt_version="matmul_v1",
            best={
                "ok": True, "attempt": 2, "cu_filename": "a.cu", "binary": "b",
                "effective_time": 0.3, "self_reported_time": 0.31, "ncu_time_sec": 0.3,
                "max_diff": float("nan"), "bottleneck": "memory", "profile_source": "ncu",
                "metrics": {"gpu__time_duration.sum": 3e8}, "feedback": "x",
            },
            records=[{
                "attempt": 1, "ok": False, "time_sec": None, "ncu_time_sec": None,
                "profile_source": None, "max_diff": None, "cu_filename": None, "binary": None,
                "out_bin": "o.bin", "bottleneck": None, "metrics": None, "feedback": None,
                "code": None, "error_feedback": "boom", "compile_stdout_tail": None,
                "compile_stderr_tail": None, "timed_out": False, "returncode": 1,
                "model": "deepseek-flash", "usage": {"prompt_tokens": 10},
            }],
            env_snapshot={"python": "3.14"},
        )
        path, index_path = eval_logic.persist_result(payload, tmp)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == "run-1"
        assert data["best"]["max_diff"] is None  # NaN -> None
        assert data["attempts"][0]["error_feedback"] == "boom"
        assert data["attempts"][0]["model"] == "deepseek-flash"
        assert os.path.exists(index_path)
        with open(index_path) as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 1
        summary = json.loads(lines[0])
        assert summary["run_id"] == "run-1" and summary["success"] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
