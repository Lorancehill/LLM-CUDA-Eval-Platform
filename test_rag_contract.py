"""rag.py 的契约与健壮性测试（v2.1 增强）。"""

import json
import os
import shutil
import tempfile
from datetime import datetime

import rag

SPEC = {
    "operator": "matmul",
    "shape": {"M": 1024, "N": 1024, "K": 1024, "batch": 1},
    "dtype": "float32",
}


def _card(cid="C1", confidence=0.5, **overrides):
    card = {
        "id": cid,
        "version": 1,
        "status": "validated",
        "operator_family": ["matmul"],
        "semantics": {"shape": {}, "dtype": "float32", "tolerance": "default"},
        "fingerprint": {},
        "transform": {"name": cid.lower(), "description": "desc",
                      "code_hint": "", "params": {}},
        "expected_gain": "1.5x",
        "failure_modes": [],
        "evidence": [],
        "confidence": confidence,
        "provenance": "test",
        "created_at": "2026-11-18",
        "last_validated": None,
    }
    card.update(overrides)
    return card


def _write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_load_cards_skips_non_object_and_corrupt(tmp_path=None):
    tmp = tempfile.mkdtemp()
    try:
        _write(os.path.join(tmp, "A.json"), _card("A"))
        _write(os.path.join(tmp, "LIST.json"), [1, 2, 3])
        with open(os.path.join(tmp, "BAD.json"), "w", encoding="utf-8") as f:
            f.write("{ not json")
        cards = rag.load_cards(tmp)
        assert [c["id"] for c in cards] == ["A"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_cards_tolerates_empty_or_none_dir():
    assert rag.load_cards(None) == []
    assert rag.load_cards("") == []
    assert rag.load_cards("/nonexistent/path/xyz") == []


def test_retrieve_tolerates_bad_arguments():
    assert rag.retrieve(None) == []
    assert rag.retrieve(SPEC, cards=None) == []
    assert rag.retrieve(SPEC, top_k=0, cards=[_card()]) == []
    assert rag.retrieve(SPEC, top_k="bad", cards=[_card()]) != []
    assert rag.retrieve(SPEC, cards=[None, "junk", _card()]) != []


def test_build_context_block_tolerates_malformed_hits():
    hits = [
        {"card": {"id": "X", "confidence": "bad",
                  "fingerprint": {"bottleneck": "memory"},
                  "transform": {"params": {"TILE": 16},
                                "failure_modes": "oops"}}},
        {"bad": 1},
        "junk",
    ]
    block = rag.build_context_block(hits)  # 不应抛异常
    assert "X" in block


def test_clamp_confidence_invalid_values():
    assert rag.clamp_confidence(None) == 0.5
    assert rag.clamp_confidence("bad") == 0.5
    assert rag.clamp_confidence(float("nan")) == 0.5
    assert rag.clamp_confidence(2.0) == rag.CONFIDENCE_MAX
    assert rag.clamp_confidence(-1.0) == rag.CONFIDENCE_MIN


def test_generalize_shape_tolerant():
    assert rag.generalize_shape(None) == {}
    assert rag.generalize_shape([1, 2]) == {}
    assert rag._generalize_dim(float("inf")) is None
    assert rag._generalize_dim("1024") == ">=512"


def test_infer_applied_cards_tolerates_string_ids():
    card = _card("TILING_01")
    card["transform"]["name"] = "shared_memory_tiling"
    prev = {"has_shared_mem": False}
    cur = {"has_shared_mem": True}
    assert rag.infer_applied_cards("TILING_01", prev, cur, [card]) == ["TILING_01"]
    assert rag.infer_applied_cards(None, None, None, None) == []
    assert rag.infer_applied_cards(["X"], {}, {}, [None, "junk", card]) == []


def test_compute_confidence_updates_tolerates_scalar_ids():
    assert rag.compute_confidence_updates("A", True) == {"A": 0.05}
    assert rag.compute_confidence_updates(None, False) == {}


def test_distill_from_run_tolerates_weird_records():
    assert rag.distill_from_run(None) is None
    assert rag.distill_from_run({"ok": True, "speedup": float("nan")}) is None
    card = rag.distill_from_run(
        {"ok": True, "speedup": "2.5", "run_id": "r1",
         "operator_spec": {"operator": "matmul", "shape": {"M": 1024, "N": 1024}}})
    assert card is not None
    assert card["evidence"][0]["speedup"] == 2.5
    assert card["semantics"]["shape"]["M"] == ">=512"


def test_load_updates_tolerates_corrupt_and_wrong_types():
    tmp = tempfile.mkdtemp()
    try:
        results = os.path.join(tmp, "results")
        run_dir = os.path.join(results, "r1")
        os.makedirs(run_dir)
        _write(os.path.join(run_dir, "knowledge_updates_16.json"),
               {"run_id": "r1", "updates": {
                   "C1": {"delta": "bad", "evidence": "oops"},
                   "C2": {"delta": 0.05, "evidence": [{"speedup": 2.0}, "junk"]},
               }})
        _write(os.path.join(run_dir, "knowledge_updates_32.json"), [1, 2])
        with open(os.path.join(run_dir, "knowledge_updates_64.json"), "w",
                  encoding="utf-8") as f:
            f.write("{bad json")
        agg = rag.load_updates(results)
        assert agg["C1"]["delta"] == 0.0
        assert agg["C2"]["delta"] == 0.05
        assert agg["C2"]["evidence"] == [{"speedup": 2.0, "run_id": "r1"}]
        assert rag.load_updates(None) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_consume_update_files_keeps_corrupt_file():
    tmp = tempfile.mkdtemp()
    try:
        results = os.path.join(tmp, "results")
        run_dir = os.path.join(results, "r1")
        os.makedirs(run_dir)
        bad = os.path.join(run_dir, "knowledge_updates_16.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{bad json")
        good = os.path.join(run_dir, "knowledge_updates_32.json")
        _write(good, {"run_id": "r1", "updates": {"C1": {"delta": 0.05}}})
        kept = rag.consume_update_files(results, [])
        assert kept == [bad]
        assert not os.path.exists(good)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_apply_updates_skips_bad_cards_and_keeps_atomic():
    tmp = tempfile.mkdtemp()
    try:
        strategies = os.path.join(tmp, "strategies")
        os.makedirs(strategies)
        _write(os.path.join(strategies, "C1.json"),
               dict(_card("C1"), evidence="oops"))  # 非 list evidence 也应能修
        with open(os.path.join(strategies, "BAD.json"), "w", encoding="utf-8") as f:
            f.write("{bad json")
        changed = rag.apply_updates(strategies, {
            "C1": {"delta": 0.05, "evidence": [{"speedup": 2.0}]},
            "BAD": {"delta": 0.05, "evidence": []},
            "GHOST": {"delta": 0.05, "evidence": []},
        })
        assert changed == ["C1"]
        with open(os.path.join(strategies, "C1.json"), encoding="utf-8") as f:
            merged = json.load(f)
        assert merged["confidence"] == 0.55
        assert merged["evidence"] == [{"speedup": 2.0}]
        assert not [n for n in os.listdir(strategies) if n.startswith(".tmp-")]
        # dry-run 不写盘
        rag.apply_updates(strategies, {"C1": {"delta": 0.05}}, dry_run=True)
        with open(os.path.join(strategies, "C1.json"), encoding="utf-8") as f:
            assert json.load(f)["confidence"] == 0.55
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_promote_candidates_skips_bad_and_is_atomic():
    tmp = tempfile.mkdtemp()
    try:
        candidates = os.path.join(tmp, "candidates")
        strategies = os.path.join(tmp, "strategies")
        os.makedirs(candidates)
        os.makedirs(strategies)
        _write(os.path.join(candidates, "SELF_A.json"),
               {"id": "SELF_A", "status": "candidate"})
        _write(os.path.join(candidates, "LIST.json"), [1, 2])
        with open(os.path.join(candidates, "BAD.json"), "w", encoding="utf-8") as f:
            f.write("{bad json")
        promoted = rag.promote_candidates(candidates, strategies)
        assert promoted == ["SELF_A"]
        dest = os.path.join(strategies, "SELF_A.json")
        assert os.path.exists(dest)
        with open(dest, encoding="utf-8") as f :
            assert json.load(f)["status"] == "curated"
        assert not os.path.exists(os.path.join(candidates, "SELF_A.json"))
        assert os.path.exists(os.path.join(candidates, "BAD.json"))
        assert not [n for n in os.listdir(strategies) if n.startswith(".tmp-")]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_update_file_has_missing_tolerates_bad_file():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "knowledge_updates_16.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{bad json")
        assert rag._update_file_has_missing(path, {"C1"}) is False
        assert rag._update_file_has_missing("/nonexistent.json", {"C1"}) is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tokenize_and_recency_tolerate_weird_input():
    assert rag._tokenize(b"ABC M=1") == ["abc"]
    assert rag._tokenize(None) == []
    score = rag._card_recency({"created_at": "2026-01-01"}, now=datetime(2026, 6, 1))
    assert 0.0 <= score <= 1.0


def test_missing_card_ids_tolerates_none_updates():
    assert rag.missing_card_ids(None, "/nonexistent") == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
