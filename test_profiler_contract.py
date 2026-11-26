"""profiler.py 的契约与健壮性测试（v2.1 增强）。"""

import shlex

import profiler

SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
L2 = "lts__t_sector_hit_rate.pct"
OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"


def _fake_ncu_csv():
    header = (
        '"ID","Process","Process ID","Kernel Name","Kernel Time","Context","Stream",'
        '"Section Name","Metric Name","Metric Unit","Metric Value"'
    )
    rows = [
        ("gpu__time_duration.sum", "123456"),
        (DRAM, "78.5"), (L2, "15.3"), (SM, "23.1"), (OCC, "31.2"),
    ]
    lines = ["Time: 0.123456 seconds", "C[0][0] = 0.000000", header]
    for name, value in rows:
        lines.append(f'0,test_1024,,matmul,123456,1,7,,{name},%,{value}')
    return "\n".join(lines)


def _csv(rows, header='"Metric Name","Metric Unit","Metric Value"'):
    body = [f"{name},%,{value}" for name, value in rows]
    return "\n".join([header] + body)


def test_metrics_contract():
    assert profiler.METRICS == [
        "gpu__time_duration.sum",
        DRAM, L2, SM, OCC,
    ]
    assert set(profiler.METRIC_LABELS) == set(profiler.METRICS)
    assert profiler.DRAM_BOUND_PCT == 60.0
    assert profiler.SM_SATURATED_PCT == 60.0


def test_build_cmd_flags_contract():
    cmd = profiler.build_ncu_cmd("test_1024")
    for token in ("--clock-control none", "--target-processes all",
                  "--launch-count 1", "--launch-skip 0",
                  "--metrics " + ",".join(profiler.METRICS), "./test_1024"):
        assert token in cmd, token
    assert cmd.startswith("ncu --csv --print-metric-name name ")


def test_parse_returns_only_known_metrics():
    m = profiler.parse_ncu_csv(_fake_ncu_csv())
    assert set(m) <= set(profiler.METRICS)
    assert all(isinstance(value, float) for value in m.values())


def test_classify_return_contract():
    result = profiler.classify_bottleneck(profiler.parse_ncu_csv(_fake_ncu_csv()))
    assert isinstance(result, tuple) and len(result) == 2
    label, feedback = result
    assert label in {"memory", "compute", "latency", "unknown"}
    assert isinstance(feedback, str) and feedback


def test_format_metrics_value_formatting():
    out = profiler.format_metrics(profiler.parse_ncu_csv(_fake_ncu_csv()))
    assert "0.123 ms" in out
    assert "78.5%" in out
    assert out.count("\n") == 4


def test_list_candidates_priority_contract():
    cands = profiler.list_ncu_candidates({"NCU_BIN": " /opt/ncu/custom "})
    assert cands[0] == "/opt/ncu/custom"
    assert cands[-1] == "ncu"


def test_classify_threshold_boundaries():
    # dram>=60 且 sm<60 → memory；sm>=60 → compute；两者都低 → latency
    assert profiler.classify_bottleneck({SM: 59.9, DRAM: 60.0})[0] == "memory"
    assert profiler.classify_bottleneck({SM: 60.0, DRAM: 60.0})[0] == "compute"
    assert profiler.classify_bottleneck({SM: 59.9, DRAM: 59.9})[0] == "latency"
    assert profiler.classify_bottleneck({SM: 60.1, DRAM: 0.0})[0] == "compute"


def test_feedback_contains_l2_and_occ_na():
    _, feedback = profiler.classify_bottleneck({SM: 20.0, DRAM: 80.0})
    assert "L2 Cache 命中率: N/A" in feedback
    assert "实际占用率: N/A" in feedback


def test_parse_accepts_none_and_bytes():
    assert profiler.parse_ncu_csv(None) == {}
    text = _csv([(SM, "85.0"), (DRAM, "30.0")])
    assert profiler.parse_ncu_csv(text.encode("utf-8"))[SM] == 85.0


def test_parse_handles_percent_suffix():
    m = profiler.parse_ncu_csv(_csv([(SM, "85.0%"), (DRAM, "30.0 %")]))
    assert m[SM] == 85.0
    assert m[DRAM] == 30.0


def test_parse_duplicate_metric_keeps_first():
    text = _csv([(SM, "85.0"), (SM, "10.0")])
    assert profiler.parse_ncu_csv(text)[SM] == 85.0


def test_parse_header_with_bom():
    text = ('\ufeff" Metric Name "," Metric Value "\n'
            f'"{SM}","85.0"\n"{DRAM}","30.0"')
    m = profiler.parse_ncu_csv(text)
    assert m[SM] == 85.0
    assert m[DRAM] == 30.0


def test_classify_non_finite_metrics_unknown():
    for bad in (float("nan"), float("inf"), float("-inf")):
        assert profiler.classify_bottleneck({SM: bad, DRAM: 80.0})[0] == "unknown"
        assert profiler.classify_bottleneck({SM: 20.0, DRAM: bad})[0] == "unknown"


def test_classify_accepts_numeric_strings():
    label, feedback = profiler.classify_bottleneck({SM: "85.0", DRAM: "30.0"})
    assert label == "compute"
    assert "计算瓶颈" in feedback


def test_format_metrics_non_finite_or_non_numeric_na():
    out = profiler.format_metrics({SM: float("nan"), DRAM: float("inf"),
                                   L2: "bad", OCC: 31.2})
    assert out.count("N/A") >= 3
    assert "nan%" not in out and "inf%" not in out


def test_build_cmd_quotes_paths_with_spaces():
    cmd = profiler.build_ncu_cmd("my binary", ncu_bin="/path with space/ncu")
    parts = shlex.split(cmd)
    assert parts[0] == "/path with space/ncu"
    assert parts[-1] == "./my binary"


def test_list_candidates_dedupes():
    cands = profiler.list_ncu_candidates({"NCU_BIN": "ncu"})
    assert cands.count("ncu") == 1


def test_classify_and_format_none_tolerated():
    assert profiler.classify_bottleneck(None)[0] == "unknown"
    assert "N/A" in profiler.format_metrics(None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
