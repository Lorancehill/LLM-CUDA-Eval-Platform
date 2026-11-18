"""profiler.py 的离线单元测试：命令构造、CSV 解析、瓶颈分类、格式化。

不依赖 GPU / Docker / ncu，用真实 ncu --csv 输出结构构造假数据验证解析与分类。
"""

import profiler


def _fake_ncu_csv():
    """构造一段带 app 噪声（Time/C[0][0]）混入的 ncu --csv 输出。"""
    header = (
        '"ID","Process","Process ID","Kernel Name","Kernel Time","Context","Stream",'
        '"Section Name","Metric Name","Metric Unit","Metric Value"'
    )
    rows = [
        # (metric_name, value)
        ("gpu__time_duration.sum", "123456"),
        ("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", "78.5"),
        ("lts__t_sector_hit_rate.pct", "15.3"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "23.1"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "31.2"),
    ]
    lines = [
        "Time: 0.123456 seconds",  # app 自身输出，混在 stdout 里
        "C[0][0] = 0.000000",
        header,
    ]
    for name, val in rows:
        lines.append(f'0,test_1024,,matmul,123456,1,7,,{name},%,{val}')
    return "\n".join(lines)


def test_build_ncu_cmd():
    cmd = profiler.build_ncu_cmd("test_1024")
    assert "ncu --csv" in cmd
    assert "--launch-count 1" in cmd
    assert "./test_1024" in cmd
    for m in profiler.METRICS:
        assert m in cmd, f"指标 {m} 应出现在命令中"


def test_build_ncu_cmd_kernel_name_filter():
    cmd = profiler.build_ncu_cmd("test_1024", kernel_name="matmul")
    assert "--kernel-name matmul" in cmd
    assert "./test_1024" in cmd
    # 不传 kernel_name 时保持「第一个 kernel launch」默认行为
    assert "--kernel-name" not in profiler.build_ncu_cmd("test_1024")


def test_parse_ncu_csv_ignores_app_noise():
    m = profiler.parse_ncu_csv(_fake_ncu_csv())
    assert m["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"] == 78.5
    assert m["lts__t_sector_hit_rate.pct"] == 15.3
    assert m["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 23.1
    assert m["sm__warps_active.avg.pct_of_peak_sustained_active"] == 31.2
    assert m["gpu__time_duration.sum"] == 123456.0
    # app 的 "Time:" 行不应被误解析进来
    assert "Time:" not in m


def test_parse_ncu_csv_empty():
    assert profiler.parse_ncu_csv("==ERROR== Failed to initialize the profiler\n") == {}


def test_classify_memory_bottleneck():
    m = profiler.parse_ncu_csv(_fake_ncu_csv())
    label, feedback = profiler.classify_bottleneck(m)
    assert label == "memory"
    assert "内存瓶颈" in feedback


def test_classify_compute_bottleneck():
    m = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
    }
    label, _ = profiler.classify_bottleneck(m)
    assert label == "compute"


def test_classify_latency_bottleneck():
    m = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
    }
    label, _ = profiler.classify_bottleneck(m)
    assert label == "latency"


def test_classify_unknown():
    label, _ = profiler.classify_bottleneck({})
    assert label == "unknown"


def test_all_bottlenecks_have_strategy_feedback():
    # 任务要求「瓶颈识别、策略选择、自动迭代」：memory/compute/latency 三类都得有策略反馈
    cases = {
        "memory": {
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        },
        "compute": {
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
        },
        "latency": {
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        },
    }
    for expected_label, m in cases.items():
        label, feedback = profiler.classify_bottleneck(m)
        assert label == expected_label
        assert isinstance(feedback, str) and len(feedback.strip()) > 0, \
            f"{expected_label} 瓶颈应返回非空策略反馈"


def test_format_metrics_handles_missing():
    out = profiler.format_metrics({})
    assert "N/A" in out


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
