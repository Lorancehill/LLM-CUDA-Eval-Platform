"""rag.py 的离线单测：加载、硬过滤、BM25 软打分、排序、渲染、自蒸馏。"""

import json
import os
import shutil
import tempfile

import rag

SPEC = {
    "operator": "matmul",
    "shape": {"M": 1024, "N": 1024, "K": 1024, "batch": 1},
    "dtype": "float32",
}


def make_card(cid="C1", family=("matmul",), bottleneck=None, source_features=None,
              shape=None, dtype="float32", confidence=0.5, ncu_ranges=None):
    card = {
        "id": cid, "version": 1, "status": "validated",
        "operator_family": list(family),
        "semantics": {"shape": shape or {}, "dtype": dtype, "tolerance": "default"},
        "fingerprint": {},
        "transform": {"name": cid.lower(), "description": "desc", "code_hint": "", "params": {}},
        "expected_gain": "1.5x", "failure_modes": [], "evidence": [],
        "confidence": confidence, "provenance": "test",
        "created_at": "2026-11-18", "last_validated": None,
    }
    if bottleneck:
        card["fingerprint"]["bottleneck"] = list(bottleneck)
    if source_features:
        card["fingerprint"]["source_features"] = dict(source_features)
    if ncu_ranges:
        card["fingerprint"]["ncu_ranges"] = dict(ncu_ranges)
    return card


def test_load_cards_loads_json():
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "a.json"), "w") as f:
            json.dump(make_card("A"), f)
        with open(os.path.join(tmp, "b.json"), "w") as f:
            json.dump(make_card("B"), f)
        cards = rag.load_cards(tmp)
        assert [c["id"] for c in cards] == ["A", "B"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_cards_missing_dir_returns_empty():
    assert rag.load_cards("/nonexistent/dir/xyz") == []


def test_eval_constraint_operators():
    assert rag._eval_constraint(1024, ">=512") is True
    assert rag._eval_constraint(256, ">=512") is False
    assert rag._eval_constraint(30, "> 40") is False
    assert rag._eval_constraint(80, "> 40") is True
    assert rag._eval_constraint(50, "< 50") is False
    assert rag._eval_constraint(1, "==1") is True
    assert rag._eval_constraint(1, "1") is True
    assert rag._eval_constraint(2, "!=1") is True
    assert rag._eval_constraint(None, ">=1") is False


def test_retrieve_filters_wrong_family():
    card = make_card("C1", family=("softmax",))
    hits = rag.retrieve(SPEC, cards=[card])
    assert hits == []


def test_retrieve_filters_shape_too_small():
    card = make_card("C1", shape={"K": ">=512"})
    assert rag.retrieve(SPEC, cards=[card]) != []  # K=1024 通过
    small = dict(SPEC, shape={"M": 256, "N": 256, "K": 256, "batch": 1})
    assert rag.retrieve(small, cards=[card]) == []  # K=256 被过滤


def test_retrieve_filters_bottleneck():
    card = make_card("C1", bottleneck=["memory"])
    assert rag.retrieve(SPEC, bottleneck="memory", cards=[card]) != []
    assert rag.retrieve(SPEC, bottleneck="compute", cards=[card]) == []


def test_retrieve_filters_source_features():
    card = make_card("C1", source_features={"has_shared_mem": False})
    assert rag.retrieve(SPEC, bottleneck=None, source_features={"has_shared_mem": False},
                        cards=[card]) != []
    assert rag.retrieve(SPEC, bottleneck=None, source_features={"has_shared_mem": True},
                        cards=[card]) == []


def test_retrieve_first_round_no_bottleneck_returns_family_cards():
    # 首轮：bottleneck=None → 不过滤瓶颈，只按族 + shape 召回
    card = make_card("C1", bottleneck=["memory"])
    hits = rag.retrieve(SPEC, bottleneck=None, cards=[card])
    assert [h["card"]["id"] for h in hits] == ["C1"]


def test_retrieve_ncu_ranges_filter():
    card = make_card("C1", ncu_ranges={"dram_util": "> 40"})
    dram_key = rag.NCU_KEY_ALIASES["dram_util"]
    assert rag.retrieve(SPEC, cards=[card], ncu_metrics={dram_key: 30.0}) == []
    assert rag.retrieve(SPEC, cards=[card], ncu_metrics={dram_key: 80.0}) != []
    # 无 ncu 指标 → ncu_ranges 跳过（不证伪）
    assert rag.retrieve(SPEC, cards=[card]) != []


def test_retrieve_top_k():
    cards = [make_card(f"C{i}") for i in range(4)]
    hits = rag.retrieve(SPEC, top_k=2, cards=cards)
    assert len(hits) == 2


def test_retrieve_sorted_by_score_and_breakdown():
    cards = [make_card("LOW", confidence=0.1), make_card("HIGH", confidence=0.9)]
    hits = rag.retrieve(SPEC, cards=cards)
    assert [h["card"]["id"] for h in hits] == ["HIGH", "LOW"]
    for h in hits:
        assert set(h["score_breakdown"].keys()) == {"bm25_norm", "prior", "confidence", "recency"}
        assert h["score_breakdown"]["bm25_norm"] == 0.0  # 首轮无瓶颈：BM25 置 0（P0-2）


def test_build_context_block_empty():
    assert "无相关" in rag.build_context_block([])


def test_build_context_block_renders_fields():
    card = make_card("TILING_01", bottleneck=["memory"], confidence=0.9)
    card["transform"]["description"] = "用 __shared__ 分块"
    card["transform"]["params"] = {"TILE": [16, 32]}
    card["transform"]["code_hint"] = "// __shared__ float As[TILE][TILE];"
    card["failure_modes"] = ["漏写 __syncthreads"]
    hits = rag.retrieve(SPEC, cards=[card])
    block = rag.build_context_block(hits)
    assert "TILING_01" in block
    assert "置信度 0.90" in block
    assert "适用瓶颈: memory" in block
    assert "失败模式" in block and "漏写 __syncthreads" in block
    assert "代码提示" in block


def test_build_context_block_truncates():
    card = make_card("BIG")
    card["transform"]["description"] = "x" * 5000
    hits = rag.retrieve(SPEC, cards=[card])
    block = rag.build_context_block(hits, max_chars=100)
    assert len(block) <= 100 + len("…（截断）\n")


def test_distill_success_returns_candidate():
    rec = {
        "ok": True, "bottleneck": "memory",
        "source_features": {"has_shared_mem": False, "has_vectorized_load": False},
        "speedup": 2.5, "run_id": "20261201-101500-ab12cd",
        "operator_spec": SPEC,
    }
    card = rag.distill_from_run(rec)
    assert card is not None
    assert card["status"] == "candidate"
    assert card["fingerprint"]["bottleneck"] == ["memory"]
    assert card["evidence"][0]["speedup"] == 2.5
    assert card["evidence"][0]["source"] == "self_experiment"
    assert card["confidence"] == 0.3


def test_distill_failure_returns_none():
    assert rag.distill_from_run({"ok": False, "speedup": 2.0}) is None


def test_distill_no_speedup_returns_none():
    rec = {"ok": True, "speedup": 0.8, "operator_spec": SPEC}
    assert rag.distill_from_run(rec) is None


def test_tokenize_filters_single_char_tokens():
    toks = rag._tokenize("M=1024 N=1024 K=1024 batch=1 float32")
    assert all(len(t) >= 2 for t in toks)
    assert "m" not in toks and "n" not in toks and "k" not in toks
    assert "1024" in toks and "float32" in toks


def test_build_query_text_has_no_single_letter_dim_tokens():
    q = rag._build_query_text(SPEC, None, None)
    assert "M=" not in q and "N=" not in q and "K=" not in q  # 不再有裸单字母维度
    assert all(len(t) >= 2 for t in rag._tokenize(q))


def test_default_strategy_dir_is_absolute_and_loads_cards():
    assert os.path.isabs(rag.DEFAULT_STRATEGY_DIR)
    cards = rag.load_cards(rag.DEFAULT_STRATEGY_DIR)
    assert len(cards) >= 5


def test_first_round_tiling_in_top2_across_sizes():
    # P0-2 回归：不同尺寸下首轮（无瓶颈）TILING 都应在 top-2，排序稳定不随尺寸漂移。
    cards = rag.load_cards(rag.DEFAULT_STRATEGY_DIR)
    for size in (256, 1024):
        spec = {"operator": "matmul",
                "shape": {"M": size, "N": size, "K": size, "batch": 1},
                "dtype": "float32"}
        hits = rag.retrieve(spec, cards=cards, top_k=3)
        ids = [h["card"]["id"] for h in hits]
        assert "TILING_01" in ids[:2], f"size={size}: TILING_01 不在首轮 top-2，实际 {ids}"


def test_bottleneck_round_uses_bm25_without_prior():
    card = make_card("C1", bottleneck=["memory"])
    hits = rag.retrieve(SPEC, bottleneck="memory", cards=[card])
    assert len(hits) == 1
    assert set(hits[0]["score_breakdown"].keys()) == {"bm25_norm", "confidence", "recency"}
    assert 0.0 <= hits[0]["score_breakdown"]["bm25_norm"] <= 1.0


def test_generalize_shape_rounds_down_pow2():
    assert rag.generalize_shape({"M": 1024, "batch": 1}) == {"M": ">=512", "batch": ">=1"}
    assert rag.generalize_shape({"M": 256, "N": 512}) == {"M": ">=128", "N": ">=256"}
    assert rag.generalize_shape({}) == {}


def test_infer_applied_cards_feature_flip():
    card = make_card("TILING_01")
    card["transform"]["name"] = "shared_memory_tiling"
    prev = {"has_shared_mem": False, "has_vectorized_load": False}
    cur = {"has_shared_mem": True, "has_vectorized_load": False}
    assert rag.infer_applied_cards(["TILING_01"], prev, cur, [card]) == ["TILING_01"]
    # 特征未翻转 → 不判 applied
    assert rag.infer_applied_cards(["TILING_01"], prev, prev, [card]) == []


def test_infer_applied_cards_register_tiling_and_unknown():
    reg = make_card("REG")
    reg["transform"]["name"] = "register_tiling"
    db = make_card("DB")
    db["transform"]["name"] = "double_buffering"
    both = {"has_shared_mem": True, "has_vectorized_load": True}
    assert rag.infer_applied_cards(["REG"], {"has_shared_mem": False}, both, [reg]) == ["REG"]
    assert rag.infer_applied_cards(["REG"], {}, {"has_shared_mem": True, "has_vectorized_load": False}, [reg]) == []
    # double_buffering 无法从静态特征可靠判断 → 跳过
    assert rag.infer_applied_cards(["DB"], {}, both, [db]) == []


def test_compute_confidence_updates_and_clamp():
    assert rag.compute_confidence_updates(["A", "B"], True) == {"A": 0.05, "B": 0.05}
    assert rag.compute_confidence_updates(["A"], False) == {"A": -0.05}
    assert rag.compute_confidence_updates([], True) == {}
    assert rag.clamp_confidence(1.2) == 0.95
    assert rag.clamp_confidence(-0.3) == 0.1
    assert rag.clamp_confidence(0.5) == 0.5


def test_distill_generalizes_shape():
    rec = {
        "ok": True, "bottleneck": "memory",
        "source_features": {"has_shared_mem": True},
        "speedup": 2.0, "run_id": "20261201-101500-ab12cd",
        "operator_spec": SPEC,
    }
    card = rag.distill_from_run(rec)
    assert card["semantics"]["shape"] == {"M": ">=512", "N": ">=512", "K": ">=512", "batch": ">=1"}
    assert card["evidence"][0]["N"] == 1024


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
