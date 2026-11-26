"""roofline.py 的契约与健壮性测试（v2.1 增强）。"""

import asyncio
import json
import os
import shutil
import tempfile

import numpy as np

import async_eval
import roofline
import validate

REQUIRED_METRIC_KEYS = {
    "profile_source",
    "timing_source",
    "flops",
    "traffic_bytes",
    "achieved_gflops",
    "achieved_gbs",
    "compute_utilization",
    "memory_utilization",
    "arithmetic_intensity",
    "ridge_point",
    "has_shared_mem",
    "has_vectorized_load",
    "has_restrict",
    "ideal_traffic_bytes",
    "roofline_bound_sec",
    "roofline_efficiency",
}


def test_public_api_surface():
    assert roofline.PEAK_FLOPS_FP32 > 0
    assert roofline.PEAK_BANDWIDTH_GBS > 0
    assert roofline.LOW_UTILIZATION == 0.005
    for name in ("matmul_flops", "estimate_traffic_bytes", "source_features",
                 "roofline_classify", "roofline_efficiency", "format_proxy_metrics"):
        assert callable(getattr(roofline, name)), name


def test_source_features_keys_and_types():
    features = roofline.source_features(None)
    assert set(features) >= {"has_shared_mem", "has_vectorized_load", "has_restrict"}
    assert all(isinstance(value, bool) for value in features.values())


def test_classify_return_contract():
    result = roofline.roofline_classify(1024, 0.1, "")
    assert isinstance(result, tuple) and len(result) == 3
    label, feedback, metrics = result
    assert label in {"memory", "compute", "latency", "unknown"}
    assert isinstance(feedback, str) and feedback
    assert isinstance(metrics, dict)


def test_metrics_required_keys():
    _, _, metrics = roofline.roofline_classify(
        1024, 0.1, "__shared__ float t[16][16];")
    missing = REQUIRED_METRIC_KEYS - set(metrics)
    assert not missing, f"metrics 缺少键: {missing}"


def test_metrics_source_fields():
    _, _, metrics = roofline.roofline_classify(1024, 0.1, "")
    assert metrics["profile_source"] == "roofline_proxy"
    assert metrics["timing_source"] == "self_reported"


def test_roofline_efficiency_formula():
    n, time_sec = 1024, 0.02
    flops = roofline.matmul_flops(n)
    ideal_traffic = 3.0 * n * n * 4
    bound = max(flops / roofline.PEAK_FLOPS_FP32,
                ideal_traffic / (roofline.PEAK_BANDWIDTH_GBS * 1e9))
    expected = min(bound / time_sec, 1.0)
    assert abs(roofline.roofline_efficiency(n, time_sec) - expected) < 1e-15
    _, _, metrics = roofline.roofline_classify(n, time_sec, "")
    assert abs(metrics["roofline_efficiency"] - expected) < 1e-15


def test_roofline_efficiency_invalid_returns_none():
    for args in [(0, 0.1), (-8, 0.1), (256, None), (256, 0.0),
                 (256, float("nan")), (256, float("inf"))]:
        assert roofline.roofline_efficiency(*args) is None, args


def test_roofline_efficiency_clamped_at_one():
    assert roofline.roofline_efficiency(256, 1e-9) == 1.0


def test_roofline_efficiency_consistent_with_metrics():
    n, time_sec = 1024, 0.02
    _, _, metrics = roofline.roofline_classify(
        n, time_sec, "__shared__ float t[16][16];")
    assert metrics["ideal_traffic_bytes"] == 3.0 * n * n * 4
    assert 0.0 <= metrics["roofline_efficiency"] <= 1.0
    assert abs(metrics["roofline_efficiency"]
               - roofline.roofline_efficiency(n, time_sec)) < 1e-15
    # 大 N 时算力下界占主导，且 shared 实现实际流量=理想流量 → 效率==算力利用率
    assert abs(metrics["roofline_efficiency"]
               - metrics["compute_utilization"]) < 1e-12


def test_format_proxy_metrics_contract():
    assert roofline.format_proxy_metrics({}) == "无有效计时，无法做 roofline 指纹。"
    _, _, metrics = roofline.roofline_classify(1024, 0.1, "")
    text = roofline.format_proxy_metrics(metrics)
    assert "GFLOP/s" in text and "GB/s" in text and "FLOP/byte" in text


def test_feedback_keywords_for_all_labels():
    cases = [
        ("memory", 1024, 0.1, ""),
        ("compute", 1024, 0.02, "__shared__ float t[16][16];"),
        ("latency", 64, 100.0, ""),
    ]
    keywords = {"memory": "内存瓶颈", "compute": "计算瓶颈", "latency": "延迟/低占用瓶颈"}
    for expected, n, time_sec, code in cases:
        label, feedback, _ = roofline.roofline_classify(n, time_sec, code)
        assert label == expected, (expected, label)
        assert keywords[expected] in feedback


def test_metrics_json_serializable():
    for n in (256, 1024, 2048):
        for time_sec in (0.02, 0.1, 100.0):
            _, _, metrics = roofline.roofline_classify(
                n, time_sec, "__shared__ float t[16][16];")
            json.dumps(metrics, allow_nan=False)


def test_numeric_string_size_supported():
    # 回归：之前 _valid_size 接受 "1024"，但后续 matmul_flops("1024") 会 TypeError
    label, _, metrics = roofline.roofline_classify("1024", 0.1, "")
    assert label in {"memory", "compute", "latency"} and metrics
    assert roofline.roofline_efficiency("1024", 0.02) is not None
    assert roofline.roofline_classify(True, 0.1, "")[0] == "unknown"


def test_zero_size_is_unknown_not_crash():
    for code in ("", "__shared__ float t[16][16];"):
        label, feedback, metrics = roofline.roofline_classify(0, 0.1, code)
        assert label == "unknown"
        assert metrics == {}
        assert isinstance(feedback, str) and feedback


def test_negative_size_is_unknown():
    label, _, metrics = roofline.roofline_classify(-8, 0.1, "")
    assert label == "unknown" and metrics == {}


def test_nan_time_is_unknown():
    label, _, metrics = roofline.roofline_classify(256, float("nan"), "")
    assert label == "unknown" and metrics == {}


def test_infinite_time_is_unknown():
    for time_sec in (float("inf"), float("-inf")):
        label, _, metrics = roofline.roofline_classify(256, time_sec, "")
        assert label == "unknown" and metrics == {}


def test_source_features_ignore_comments_and_strings():
    code = '// 这里提到 __shared__ 和 float4，但只是注释\nint x; const char* s = "__shared__";'
    features = roofline.source_features(code)
    assert features["has_shared_mem"] is False
    assert features["has_vectorized_load"] is False
    real = roofline.source_features("__shared__ float s[16]; float4 v;")
    assert real["has_shared_mem"] is True and real["has_vectorized_load"] is True


def test_source_features_detects_more_vector_types():
    for code in ("float2 v;", "int4 v;", "half4 v;", "double2 v;"):
        assert roofline.source_features(code)["has_vectorized_load"] is True, code


# ---------- 与 async_eval 的降级集成：ncu 失败时走 roofline ----------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None
        self.model = "fake-model"


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, **kwargs):
        return self._responses.pop(0)


def _fake_client(responses):
    chat = type("Chat", (), {"completions": _FakeCompletions(responses)})()
    return type("Client", (), {"chat": chat})()


def _fake_docker_run(n):
    async def fake(cmd, container_name, timeout=None):
        ref = validate.reference_matmul(n)
        ref.astype(np.float32).tofile(f"cuda_output_{n}.bin")
        return 0, b"Time: 0.5 seconds\n", b"", False

    return fake


async def _fake_profile(matrix_size, binary, kernel_name=None):
    return None


def test_async_eval_falls_back_to_roofline():
    tmp = tempfile.mkdtemp()
    old_cwd = os.getcwd()
    saved = {}
    try:
        os.chdir(tmp)
        saved = {
            "client": async_eval.client,
            "_docker_run": async_eval._docker_run,
            "_profile_kernel": async_eval._profile_kernel,
            "RESULTS_DIR": async_eval.RESULTS_DIR,
            "_knowledge_config": async_eval._knowledge_config,
        }
        async_eval.client = _fake_client([_FakeResponse("__global__ void k(){}")] * 3)
        async_eval._docker_run = _fake_docker_run(16)
        async_eval._profile_kernel = _fake_profile
        async_eval.RESULTS_DIR = os.path.join(tmp, "results")
        async_eval._knowledge_config = lambda: {"enabled": False, "top_k": 3,
                                                "mode": "cards"}

        ok, _, _ = asyncio.run(
            async_eval.evaluate_matrix(16, "run-roofline", {"python": "test"}))
        assert ok is True

        path = os.path.join(tmp, "results", "run-roofline", "16.json")
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        first = payload["attempts"][0]
        assert first["profile_source"] == "roofline_proxy"
        assert first["bottleneck"] in {"memory", "compute", "latency"}
        assert first["metrics"]["profile_source"] == "roofline_proxy"
        assert first["metrics"]["timing_source"] == "self_reported"
        assert payload["best"]["profile_source"] == "roofline_proxy"
    finally:
        for key, value in saved.items():
            setattr(async_eval, key, value)
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
