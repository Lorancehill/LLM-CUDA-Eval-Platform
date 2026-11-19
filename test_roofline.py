"""roofline.py 的离线单测：算子规格、流量估算、源码特征、瓶颈分类。"""

import roofline


def test_matmul_flops():
    assert roofline.matmul_flops(10) == 2000.0


def test_estimate_traffic_naive():
    N = 8
    expected = 2 * N ** 3 * 4 + N * N * 4
    assert roofline.estimate_traffic_bytes(N, has_shared_mem=False) == expected


def test_estimate_traffic_shared():
    assert roofline.estimate_traffic_bytes(8, has_shared_mem=True) == 3 * 8 * 8 * 4


def test_source_features():
    code = "__shared__ float s[16]; float4 v; __restrict__ float* p;"
    f = roofline.source_features(code)
    assert f["has_shared_mem"] and f["has_vectorized_load"] and f["has_restrict"]


def test_roofline_memory_bound_naive():
    # naive（无 __shared__）：带宽利用率 >> 算力利用率 → memory
    N = 1024
    label, feedback, m = roofline.roofline_classify(N, time_sec=0.1, code="")
    assert label == "memory"
    assert m["profile_source"] == "roofline_proxy"
    assert "内存瓶颈" in feedback


def test_roofline_compute_bound_tiled():
    # 有 __shared__（分块复用）且较快：算力利用率应高于带宽利用率 → compute
    N = 1024
    label, feedback, _ = roofline.roofline_classify(
        N, time_sec=0.02, code="__shared__ float tile[16][16];"
    )
    assert label == "compute"
    assert "计算瓶颈" in feedback


def test_roofline_latency_bound():
    # 极慢（时间很长）→ 两者利用率都极低 → latency
    label, _, _ = roofline.roofline_classify(64, time_sec=100.0, code="")
    assert label == "latency"


def test_roofline_unknown_without_time():
    label, feedback, m = roofline.roofline_classify(256, None, "")
    assert label == "unknown"
    assert m == {}
    assert "无可用计时" in feedback


def test_roofline_clamps_utilization_at_100():
    # 自报计时极快 → 利用率 >100%（如 207%），应夹到 1.0（P1）
    label, feedback, m = roofline.roofline_classify(
        256, time_sec=1e-9, code="__shared__ float t[16][16];"
    )
    assert m["compute_utilization"] == 1.0
    assert m["memory_utilization"] == 1.0
    assert "100.00%" in roofline.format_proxy_metrics(m)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
