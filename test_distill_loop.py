"""R3 自蒸馏 + 归因闭环注入测试：假 client + 假 profiler + 假沙箱。

验证：
- 两轮（第二轮含 __shared__ 且更快）→ applied_cards 归因到 TILING_01、
  speedup_vs_prev > 1、生成候选卡文件、置信度 +0.05；
- 更慢轮 → 置信度 -0.05 且不生成候选卡；
- 编译失败轮 → 置信度 -0.05 且不生成候选卡。

不依赖 GPU / Docker / ncu / API。
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


def _monkeypatch(**overrides):
    saved = {k: getattr(async_eval, k) for k in overrides}
    for k, v in overrides.items():
        setattr(async_eval, k, v)
    return saved


def _memory_metrics(ns):
    return {
        "gpu__time_duration.sum": float(ns),
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
    }


def _make_tiling_card():
    return {
        "id": "TILING_01", "version": 1, "status": "validated",
        "operator_family": ["matmul"],
        "semantics": {"shape": {}, "dtype": "float32", "tolerance": "default"},
        "fingerprint": {"bottleneck": ["memory"], "source_features": {"has_shared_mem": False}},
        "transform": {
            "name": "shared_memory_tiling",
            "description": "用 __shared__ 分块", "code_hint": "", "params": {"TILE": [16, 32]},
        },
        "expected_gain": "2x", "failure_modes": [], "evidence": [],
        "confidence": 0.6, "provenance": "test", "created_at": "2026-11-18", "last_validated": None,
    }


def _setup(tmp, responses, times, metrics_sequence, fail_attempts=(), N=16):
    os.makedirs(os.path.join(tmp, "strategies"), exist_ok=True)
    with open(os.path.join(tmp, "strategies", "TILING_01.json"), "w", encoding="utf-8") as f:
        json.dump(_make_tiling_card(), f)
    return _monkeypatch(
        RESULTS_DIR=os.path.join(tmp, "results"),
        STRATEGY_DIR=os.path.join(tmp, "strategies"),
        CANDIDATES_DIR=os.path.join(tmp, "knowledge", "candidates"),
        client=_fake_client(responses),
        _docker_run=_make_fake_docker_run(times, N, fail_attempts),
        _profile_kernel=_make_fake_profile(metrics_sequence),
        _knowledge_config=lambda: {"enabled": True, "top_k": 3, "mode": "cards"},
    )


def _restore(saved, old_cwd):
    if saved:
        for k, v in saved.items():
            setattr(async_eval, k, v)
    os.chdir(old_cwd)


def test_distill_success_loop():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        # 第二轮 code 含 __shared__ 且更快；第二轮画像只给耗时 → label='unknown' → 循环在 2 轮结束
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code with tiling\n__shared__ float As[16][16];"),
            ],
            times=[0.5, 0.1],
            metrics_sequence=[_memory_metrics(5.0e8), {"gpu__time_duration.sum": 1.0e8}],
        )

        ok, _timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(16, "run-distill", {"python": "test"}))
        assert ok is True

        json_path = os.path.join(tmp, "results", "run-distill", "16.json")
        with open(json_path) as f:
            payload = json.load(f)
        attempts = payload["attempts"]
        assert len(attempts) == 2
        assert attempts[0]["applied_cards"] == []           # 首轮无上一轮检索卡
        assert attempts[0]["speedup_vs_prev"] is None       # 首轮无 prev_time
        assert attempts[0]["source_features"]["has_shared_mem"] is False
        assert attempts[1]["applied_cards"] == ["TILING_01"]
        assert attempts[1]["speedup_vs_prev"] > 1
        assert attempts[1]["source_features"]["has_shared_mem"] is True

        # P1-1：提速来自已应用卡 TILING_01 → 不蒸馏重复候选卡，只追加 evidence
        assert payload["knowledge"]["candidates_created"] == []
        assert not os.path.exists(os.path.join(tmp, "knowledge", "candidates"))

        # 置信度 +0.05 且带 evidence（P1-4）；文件名带尺寸（P0-1）
        updates_path = os.path.join(tmp, "results", "run-distill", "knowledge_updates_16.json")
        with open(updates_path) as f:
            updates = json.load(f)
        upd = updates["updates"]["TILING_01"]
        assert upd["delta"] == 0.05
        assert len(upd["evidence"]) == 1
        assert upd["evidence"][0]["speedup"] > 1
        assert payload["knowledge"]["knowledge_updates_applied"]["TILING_01"]["delta"] == 0.05
    finally:
        _restore(saved, old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_distill_slower_round_penalizes():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code with tiling but slower\n__shared__ float As[16][16];"),
            ],
            times=[0.1, 0.5],
            metrics_sequence=[_memory_metrics(1.0e8), {"gpu__time_duration.sum": 5.0e8}],
        )

        ok, _timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(16, "run-slower", {"python": "test"}))
        assert ok is True

        json_path = os.path.join(tmp, "results", "run-slower", "16.json")
        with open(json_path) as f:
            payload = json.load(f)
        attempts = payload["attempts"]
        assert attempts[1]["applied_cards"] == ["TILING_01"]
        assert attempts[1]["speedup_vs_prev"] < 1

        # 无候选卡（best 不比第一次 attempt 快）
        cand_dir = os.path.join(tmp, "knowledge", "candidates")
        assert not os.path.exists(cand_dir)
        assert payload["knowledge"]["candidates_created"] == []

        updates_path = os.path.join(tmp, "results", "run-slower", "knowledge_updates_16.json")
        with open(updates_path) as f:
            updates = json.load(f)
        assert updates["updates"]["TILING_01"]["delta"] == -0.05
    finally:
        _restore(saved, old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_distill_compile_failure_penalizes():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code bad"),
                _FakeResponse("// code naive again"),
            ],
            times=[0.5, None, 0.5],
            metrics_sequence=[_memory_metrics(5.0e8), _memory_metrics(5.0e8)],
            fail_attempts=(2,),
        )

        ok, timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(16, "run-compilefail", {"python": "test"}))
        assert ok is True                      # 第 1 轮成功，best 保优
        assert timings == [0.5, None, 0.5]

        json_path = os.path.join(tmp, "results", "run-compilefail", "16.json")
        with open(json_path) as f:
            payload = json.load(f)
        attempts = payload["attempts"]
        assert attempts[1]["ok"] is False
        assert attempts[1]["applied_cards"] == []     # 编译失败轮无法归因成功应用

        assert payload["knowledge"]["candidates_created"] == []

        updates_path = os.path.join(tmp, "results", "run-compilefail", "knowledge_updates_16.json")
        with open(updates_path) as f:
            updates = json.load(f)
        assert updates["updates"]["TILING_01"]["delta"] == -0.05
    finally:
        _restore(saved, old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_distill_new_card_when_no_existing_card():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        # 第二轮更快，但未翻转任何已知卡的源码特征（无 __shared__/float4）→ applied_cards=[]，
        # 说明模型用了知识库未覆盖的策略 → 应蒸馏新候选卡，且 id 含尺寸与 attempt（P1-2）。
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code faster via restrict\n__restrict__ float* a;"),
            ],
            times=[0.5, 0.2],
            metrics_sequence=[_memory_metrics(5.0e8), {"gpu__time_duration.sum": 2.0e8}],
        )

        ok, _timings, _ncu_timings = asyncio.run(async_eval.evaluate_matrix(16, "run-newcard", {"python": "test"}))
        assert ok is True

        json_path = os.path.join(tmp, "results", "run-newcard", "16.json")
        with open(json_path) as f:
            payload = json.load(f)
        assert payload["attempts"][1]["applied_cards"] == []
        assert payload["knowledge"]["candidates_created"] == ["SELF_runnewcard-16-a2"]

        cand_path = os.path.join(tmp, "knowledge", "candidates", "SELF_runnewcard-16-a2.json")
        assert os.path.exists(cand_path)
        with open(cand_path) as f:
            cand = json.load(f)
        assert cand["id"] == "SELF_runnewcard-16-a2"
    finally:
        _restore(saved, old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_concurrent_sizes_write_distinct_update_files():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = None
    try:
        os.chdir(tmp)
        # 尺寸 16：第二轮应用 TILING → 写 knowledge_updates_16.json
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code with tiling\n__shared__ float As[16][16];"),
            ],
            times=[0.5, 0.1],
            metrics_sequence=[_memory_metrics(5.0e8), {"gpu__time_duration.sum": 1.0e8}],
            N=16,
        )
        asyncio.run(async_eval.evaluate_matrix(16, "run-concurrent", {"python": "test"}))
        _restore(saved, old_cwd)

        # 尺寸 32：同一 run_id、同样第二轮应用 TILING → 写 knowledge_updates_32.json
        saved = _setup(
            tmp,
            responses=[
                _FakeResponse("// code naive"),
                _FakeResponse("// code with tiling\n__shared__ float As[16][16];"),
            ],
            times=[0.5, 0.1],
            metrics_sequence=[_memory_metrics(5.0e8), {"gpu__time_duration.sum": 1.0e8}],
            N=32,
        )
        asyncio.run(async_eval.evaluate_matrix(32, "run-concurrent", {"python": "test"}))
        _restore(saved, old_cwd)

        # P0-1：两个尺寸各一份、互不覆盖
        upd16 = os.path.join(tmp, "results", "run-concurrent", "knowledge_updates_16.json")
        upd32 = os.path.join(tmp, "results", "run-concurrent", "knowledge_updates_32.json")
        assert os.path.exists(upd16)
        assert os.path.exists(upd32)
        with open(upd16) as f:
            assert json.load(f)["matrix_size"] == 16
        with open(upd32) as f:
            assert json.load(f)["matrix_size"] == 32
    finally:
        _restore(saved, old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
