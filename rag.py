"""RAG 知识库：策略卡（Strategy Card）加载、检索、渲染与自蒸馏（MVP，零新依赖）。

设计依据《RAG知识库设计与实施计划.md》§1/§3：
- 知识表示：knowledge/strategies/*.json，一张卡 = 一个可执行优化策略 + 适用条件 + 证据。
- 检索不是纯文本相似，而是「硬过滤（算子族 + 瓶颈指纹 + 语义约束）+ 软打分（BM25）+
  先验加权（置信度 + 近因）」。
- 全部纯函数，离线可单测；不 import async_eval（它 import 即创建 API client）。

卡片 schema（对应设计文档 §1 的 YAML，改用 JSON 以保持零新依赖）：
  id, version, status, operator_family, semantics{shape,dtype,tolerance},
  fingerprint{bottleneck, source_features, ncu_ranges?},
  transform{name, description, code_hint, params},
  expected_gain, failure_modes, evidence, confidence, provenance, created_at, last_validated
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from datetime import datetime, timezone

# ---- 软打分权重：score = w1*bm25_norm + w2*confidence + w3*recency ----
W_BM25 = 0.5
W_CONFIDENCE = 0.35
W_RECENCY = 0.15

# 近因衰减半衰期（天）：last_validated/created_at 距今这么久后 recency 衰减到 0.5
RECENCY_HALF_LIFE_DAYS = 180.0

# Okapi BM25 参数
BM25_K1 = 1.5
BM25_B = 0.75

# 卡片 ncu_ranges 的短键 → profiler.METRICS 完整指标名
NCU_KEY_ALIASES = {
    "dram_util": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_util": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit": "lts__t_sector_hit_rate.pct",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CONSTRAINT_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<|=)?\s*(-?\d+(?:\.\d+)?)\s*$")


def _eval_constraint(value, expr) -> bool:
    """求值单个约束表达式（semantics.shape / ncu_ranges 共用），返回 bool。

    支持 ">=512" "> 40" "< 50" "<=60" "==1" "=1" "!=0" 及裸值（相等）。
    value 为 None（未知）一律 False（未知不满足硬过滤，从严，避免召回噪声）。
    """
    if value is None:
        return False
    expr = str(expr).strip()
    m = _CONSTRAINT_RE.match(expr)
    if not m:
        # 非数值表达式（如字符串枚举）：退化为相等比较
        return str(value) == expr
    op = m.group(1) or "=="
    try:
        target = float(m.group(2))
        value = float(value)
    except (TypeError, ValueError):
        return False
    if op in ("=", "=="):
        return value == target
    if op == "!=":
        return value != target
    if op == ">":
        return value > target
    if op == "<":
        return value < target
    if op == ">=":
        return value >= target
    if op == "<=":
        return value <= target
    return False


def load_cards(card_dir: str) -> list[dict]:
    """加载 card_dir 下所有 *.json 策略卡，按路径排序返回；目录缺失返回 []。"""
    cards = []
    for path in sorted(glob.glob(os.path.join(card_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            cards.append(json.load(f))
    return cards


def _tokenize(text: str) -> list[str]:
    """英文/数字/下划线分词（MVP 不做中文分词：查询与卡片的英文字段即可匹配）。"""
    return _TOKEN_RE.findall((text or "").lower())


def _operator_family_match(card: dict, operator_spec: dict) -> bool:
    family = operator_spec.get("operator") or operator_spec.get("family")
    if not family:
        return True  # 查询未给算子族 → 不按族过滤
    card_fams = card.get("operator_family") or []
    if not card_fams:
        return True  # 卡片未声明族 → 视为通配
    return family in card_fams


def _semantics_satisfied(card: dict, operator_spec: dict) -> bool:
    sem = card.get("semantics") or {}
    # dtype：严格相等（缺省视为不约束）
    if sem.get("dtype") is not None and operator_spec.get("dtype") is not None:
        if str(sem["dtype"]) != str(operator_spec["dtype"]):
            return False
    # shape 约束：只对查询提供的维度求值；查询缺维则宽进（不证伪）
    shape = operator_spec.get("shape") or {}
    for dim, expr in (sem.get("shape") or {}).items():
        if dim in shape and not _eval_constraint(shape[dim], expr):
            return False
    return True


def _fingerprint_satisfied(card: dict, bottleneck, source_features, ncu_metrics) -> bool:
    fp = card.get("fingerprint") or {}
    # 瓶颈：查询给了 bottleneck 且卡片声明了 bottleneck 列表时才要求命中
    if bottleneck is not None:
        card_bn = fp.get("bottleneck")
        if card_bn and bottleneck not in card_bn:
            return False
    # 源码特征：查询提供且卡片约束了某特征时要求一致；查询缺该特征则宽进
    if source_features:
        for key, expected in (fp.get("source_features") or {}).items():
            if key in source_features and bool(source_features[key]) != bool(expected):
                return False
    # ncu 精确条件（可选）：查询提供指标且卡片约束时才求值
    if ncu_metrics:
        for key, expr in (fp.get("ncu_ranges") or {}).items():
            metric_name = NCU_KEY_ALIASES.get(key, key)
            if metric_name in ncu_metrics and not _eval_constraint(ncu_metrics[metric_name], expr):
                return False
    return True


def _build_query_text(operator_spec, bottleneck, source_features) -> str:
    parts = [operator_spec.get("operator") or operator_spec.get("family") or ""]
    shape = operator_spec.get("shape") or {}
    if shape:
        parts.append(" ".join(f"{k}={v}" for k, v in sorted(shape.items())))
    if operator_spec.get("dtype"):
        parts.append(str(operator_spec["dtype"]))
    if bottleneck:
        parts.append(f"bottleneck {bottleneck}")
    for k, v in sorted((source_features or {}).items()):
        parts.append(f"{k}={v}")
    return " ".join(p for p in parts if p)


def _card_doc_text(card: dict) -> str:
    t = card.get("transform") or {}
    fp = card.get("fingerprint") or {}
    parts = [
        card.get("id", ""),
        t.get("name", ""),
        t.get("description", ""),
        card.get("provenance", ""),
        " ".join(card.get("operator_family") or []),
        " ".join(fp.get("bottleneck") or []),
        " ".join((fp.get("source_features") or {}).keys()),
        " ".join((t.get("params") or {}).keys()),
        " ".join(card.get("failure_modes") or []),
        str((card.get("semantics") or {}).get("dtype") or ""),
    ]
    return " ".join(parts)


def _bm25(query_terms: list[str], docs: list[list[str]]) -> list[float]:
    """Okapi BM25 打分。docs 为 token 列表的列表，返回与 docs 等长的分数列表。"""
    n = len(docs)
    if n == 0:
        return []
    df: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    avgdl = sum(len(d) for d in docs) / n
    scores = []
    for doc in docs:
        dl = len(doc)
        tf: dict[str, int] = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            f = tf[term]
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (f * (BM25_K1 + 1.0)) / (
                f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl)
            )
        scores.append(score)
    return scores


def _card_recency(card: dict, now: datetime | None = None) -> float:
    """近因得分（0~1）：按 last_validated（缺则 created_at）距今的天数指数衰减。"""
    date_str = card.get("last_validated") or card.get("created_at")
    if not date_str:
        return 0.5
    try:
        d = datetime.fromisoformat(str(date_str))
    except ValueError:
        return 0.5
    now = now or datetime.now(timezone.utc)
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - d).total_seconds() / 86400.0)
    return 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)


def retrieve(operator_spec, bottleneck=None, source_features=None, ncu_metrics=None,
             top_k=3, cards=None):
    """检索策略卡，返回按综合分降序的 top_k 条 [{card, score, score_breakdown}, ...]。

    硬过滤：算子族 + semantics（dtype/shape）+ fingerprint（bottleneck/source_features/ncu_ranges）。
    软打分：score = w1*bm25_norm + w2*confidence + w3*recency，
    bm25 对「算子名 + 形状 + 瓶颈标签 + 源码特征」组成的查询文本打分。
    bottleneck 为 None 时跳过瓶颈过滤（首轮注入场景：尚未画像）。
    """
    source_features = source_features or {}
    scored = []
    for card in (cards or []):
        if not _operator_family_match(card, operator_spec):
            continue
        if not _semantics_satisfied(card, operator_spec):
            continue
        if not _fingerprint_satisfied(card, bottleneck, source_features, ncu_metrics):
            continue
        scored.append(card)
    if not scored:
        return []
    query_terms = _tokenize(_build_query_text(operator_spec, bottleneck, source_features))
    bm25_scores = _bm25(query_terms, [_tokenize(_card_doc_text(c)) for c in scored])
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    hits = []
    for card, bm in zip(scored, bm25_scores):
        bm25_norm = (bm / max_bm25) if max_bm25 > 0 else 0.0
        conf = float(card.get("confidence", 0.5))
        rec = _card_recency(card)
        hits.append({
            "card": card,
            "score": W_BM25 * bm25_norm + W_CONFIDENCE * conf + W_RECENCY * rec,
            "score_breakdown": {
                "bm25_norm": bm25_norm,
                "confidence": conf,
                "recency": rec,
            },
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:top_k]


def build_context_block(hits, max_chars=3000) -> str:
    """把检索结果渲染成注入 prompt 的知识块（id/适用条件/参数/预期收益/失败模式/代码提示）。"""
    if not hits:
        return "[知识库检索结果] 无相关策略卡。"
    lines = [f"[知识库检索结果（按相关度排序，共 {len(hits)} 条）]"]
    for h in hits:
        c = h["card"]
        t = c.get("transform") or {}
        bn = (c.get("fingerprint") or {}).get("bottleneck") or []
        lines.append(
            f"- [{c.get('id')}, 置信度 {float(c.get('confidence', 0.5)):.2f}] "
            f"适用瓶颈: {', '.join(bn) if bn else '任意'}"
        )
        if t.get("description"):
            lines.append(f"  做法: {t['description']}")
        params = t.get("params") or {}
        if params:
            lines.append("  参数: " + ", ".join(
                f"{k}∈{{{', '.join(map(str, v))}}}" for k, v in params.items()))
        if c.get("expected_gain"):
            lines.append(f"  预期收益: {c['expected_gain']}")
        fm = c.get("failure_modes") or []
        if fm:
            lines.append("  失败模式: " + "; ".join(fm))
        if t.get("code_hint"):
            lines.append("  代码提示:\n" + t["code_hint"])
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n…（截断）"
    return block


def _candidate_card_id(run_record: dict) -> str:
    rid = str(run_record.get("run_id") or "")
    clean = re.sub(r"[^A-Za-z0-9]", "", rid)
    return ("SELF_" + clean[:12]) if clean else "SELF_UNKNOWN"


def distill_from_run(run_record: dict) -> dict | None:
    """从一轮「成功且更快」的评测结果蒸馏一张候选卡（status='candidate'）。

    run_record 需含：ok、bottleneck、source_features、speedup、run_id、operator_spec。
    - ok=True 且 speedup>1 → 返回候选卡 dict（fingerprint 取自本轮瓶颈/源码特征，
      evidence 记录 run_id/speedup，confidence 初值 0.3）。
    - 其余（失败/未更快）→ None。失败/变慢的 failure_modes 更新需对比前后代码的
      LLM 总结（R3 阶段），此处不做，避免无依据地污染负知识。
    """
    if not run_record.get("ok"):
        return None
    speedup = run_record.get("speedup")
    if speedup is None or speedup <= 1.0:
        return None
    op = run_record.get("operator_spec") or {}
    bottleneck = run_record.get("bottleneck")
    feats = run_record.get("source_features") or {}
    shape = op.get("shape") or {}
    fingerprint = {}
    if bottleneck:
        fingerprint["bottleneck"] = [bottleneck]
    if feats:
        fingerprint["source_features"] = {k: bool(v) for k, v in feats.items()}
    return {
        "id": _candidate_card_id(run_record),
        "version": 1,
        "status": "candidate",
        "operator_family": [op.get("operator") or op.get("family") or "matmul"],
        "semantics": {
            "shape": shape,
            "dtype": op.get("dtype", "float32"),
            "tolerance": "default",
        },
        "fingerprint": fingerprint,
        "transform": {
            "name": "self_distilled",
            "description": (
                f"自蒸馏候选：{bottleneck or 'unknown'} 瓶颈下 speedup={speedup:.2f}x；"
                f"具体代码见 results/{run_record.get('run_id')} 对应 attempt 的 code 全文。"
            ),
            "code_hint": "",
            "params": {},
        },
        "expected_gain": f"{speedup:.2f}x",
        "failure_modes": [],
        "evidence": [{
            "run_id": run_record.get("run_id"),
            "source": "self_experiment",
            "speedup": speedup,
            "N": shape.get("N"),
        }],
        "confidence": 0.3,
        "provenance": "自身实验蒸馏（self-distillation）",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "last_validated": None,
    }
