"""闭环注入测试：假 client + 假 profiler + 假沙箱跑通 best-so-far / 三类瓶颈反馈 / 落盘。

不依赖 GPU / Docker / ncu / API：monkeypatch async_eval 的协作点，验证
「瓶颈识别 → 策略反馈 → 继续优化 → best-so-far 保优 → 证据链落盘」的纯逻辑闭环。
"""

import asyncio
import json
import os
import shutil
import tempfile

import numpy as np

import async_eval
import validate


class _FakeResponse:
    def __init__(self, content, model="deepseek-flash", usage=None):
        self.choices = [type("M", (), {"message": type("Msg", (), {"content": content})()})()]
        self.model = model
        self.usage = usage


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, **kwargs):
        return self._responses.pop(0)


def _fake_client(responses):
    return type("C", (), {"chat": type("Chat", (), {"completions": _FakeCompletions(responses)})()})()


def _make_fake_docker_run(times, N, fail_attempts=()):
    """按调用序号返回对应 Time 与正确矩阵；fail_attempts（1 起）模拟改坏的内核。"""
    counter = {"n": 0}

    async def fake(cmd, container_name, timeout=None):
        counter["n"] += 1
        idx = counter["n"]
        if idx in fail_attempts:
            return 1, b"", b"error: compilation failed", False
        ref = validate.reference_matmul(N)
        ref.astype(np.float32).tofile(f"cuda_output_{N}.bin")
        t = times[idx - 1]
        return 0, f"Time: {t} seconds\nC[0][0] = 0.000000".encode(), b"", False

    return fake


def _make_fake_profile(metrics_sequence):
    counter = {"n": 0}

    async def fake(matrix_size, binary, kernel_name=None):
        counter["n"] += 1
        return metrics_sequence[counter["n"] - 1]

    return fake


def _monkeypatch(tmp, **overrides):
    saved = {k: getattr(async_eval, k) for k in overrides}
    for k, v in overrides.items():
        setattr(async_eval, k, v)
    return saved


def test_loop_three_bottlenecks_pick_fastest():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        N = 16
        times = [0.5, 0.3, 0.1]
        saved = _monkeypatch(
            tmp,
            RESULTS_DIR=os.path.join(tmp, "results"),
            client=_fake_client([
                _FakeResponse("// code naive"),
                _FakeResponse("// code compute"),
                _FakeResponse("// code latency"),
            ]),
            _docker_run=_make_fake_docker_run(times, N),
            _profile_kernel=_make_fake_profile([
                {"gpu__time_duration.sum": 5.0e8,
                 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
                 "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0},
                {"gpu__time_duration.sum": 3.0e8,
                 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
                 "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0},
                {"gpu__time_duration.sum": 1.0e8,
                 "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
                 "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0},
            ]),
        )

        ok, timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(N, "run-test", {"python": "test"}))
        assert ok is True
        assert timings == times

        json_path = os.path.join(tmp, "results", "run-test", f"{N}.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["success"] is True
        assert payload["best"]["attempt"] == 3
        assert payload["best"]["effective_time"] == 0.1
        assert [a["bottleneck"] for a in payload["attempts"]] == ["memory", "compute", "latency"]
        assert all(a["feedback"] for a in payload["attempts"])

        index_path = os.path.join(tmp, "results", "index.jsonl")
        assert os.path.exists(index_path)
        with open(index_path) as f:
            assert len(f.read().strip().splitlines()) == 1
    finally:
        if saved:
            for k, v in saved.items():
                setattr(async_eval, k, v)
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_loop_failure_does_not_kill_best():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        N = 16
        saved = _monkeypatch(
            tmp,
            RESULTS_DIR=os.path.join(tmp, "results"),
            client=_fake_client([
                _FakeResponse("// code good"),
                _FakeResponse("// code bad"),
                _FakeResponse("// code bad"),
            ]),
            _docker_run=_make_fake_docker_run([0.5, 0.2, 0.2], N, fail_attempts=(2, 3)),
            _profile_kernel=_make_fake_profile([None, None, None]),
        )

        ok, timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(N, "run-test2", {"python": "test"}))
        assert ok is True
        assert timings == [0.5, None, None]

        json_path = os.path.join(tmp, "results", "run-test2", f"{N}.json")
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["best"]["attempt"] == 1
        assert payload["best"]["effective_time"] == 0.5
        assert payload["attempts"][1]["ok"] is False
        assert payload["attempts"][1]["error_feedback"]
    finally:
        if saved:
            for k, v in saved.items():
                setattr(async_eval, k, v)
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
