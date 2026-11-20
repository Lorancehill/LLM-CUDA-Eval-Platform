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

import argparse
import glob
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone

# ---- 软打分权重：score = w1*bm25_norm + w2*confidence + w3*recency ----
W_BM25 = 0.5
W_CONFIDENCE = 0.35
W_RECENCY = 0.15

# 首轮（bottleneck=None）没有真实检索信号，用「策略先验」替代 BM25（P0-2）。
# 权重与 W_BM25 同位（替换而非叠加）。
W_PRIOR = 0.5

# 策略先验（按 transform.name）：无瓶颈画像时按「基础 → 进阶」的通用性排序，
# 保证同一策略在不同尺寸下首轮注入顺序稳定。未列出的 transform 用默认 0.5。
STRATEGY_PRIOR = {
    "shared_memory_tiling": 0.9,
    "vectorized_load": 0.8,
    "split_k": 0.7,
    "register_tiling": 0.6,
    "double_buffering": 0.5,
}

# 近因衰减半衰期（天）：last_validated/created_at 距今这么久后 recency 衰减到 0.5
RECENCY_HALF_LIFE_DAYS = 180.0

# R3 自蒸馏：每张应用卡的置信度增量与上下限（人工 review 后合并时夹取）
CONFIDENCE_STEP = 0.05
CONFIDENCE_MIN = 0.1
CONFIDENCE_MAX = 0.95

# 归因签名：由「前后两轮源码特征变化」推断模型是否应用了某张检索到的卡。
# tuple(feature, prev_expected, cur_expected) = 特征从 prev_expected 翻转到 cur_expected；
# None 表示无法只靠单一特征翻转判断（register_tiling 需 shared+vectorized 同时为 true，
# double_buffering/split_k 无法从静态特征可靠判断，标记 unknown 跳过）。
CARD_FEATURE_SIGNATURES = {
    "shared_memory_tiling": ("has_shared_mem", False, True),
    "vectorized_load":     ("has_vectorized_load", False, True),
    "register_tiling":     None,
    "double_buffering":    None,
    "split_k":             None,
}

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


# 默认策略卡目录：基于本文件绝对路径（而非进程 cwd），
# 换工作目录/定时任务/绝对路径运行时不会静默加载不到卡（P0-1）。
DEFAULT_STRATEGY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "knowledge", "strategies"
)


def load_cards(card_dir: str) -> list[dict]:
    """加载 card_dir 下所有 *.json 策略卡，按路径排序返回；目录缺失返回 []。

    单张卡 JSON 损坏时跳过并告警，而不是让整批评测崩溃（merge 中断/手工编辑都可能留下
    半截文件；坏卡跳过等价于「该知识不可用」，其余卡照常检索）。
    """
    cards = []
    for path in sorted(glob.glob(os.path.join(card_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                cards.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️ 跳过无法加载的策略卡 {path}: {e}")
    return cards


def _tokenize(text: str) -> list[str]:
    """英文/数字/下划线分词（MVP 不做中文分词：查询与卡片的英文字段即可匹配）。

    过滤长度 <2 的 token：M/N/K/1 等单字符是 BM25 噪声，会匹配卡片文本里的杂散字母
    并扭曲排序（P0-2）。
    """
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 2]


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
        # 维度名加 dim_ 前缀并转小写，避免 M=/N=/K= 产生单字母 token（P0-2）；
        # 残余单字母再由 _tokenize 的 len>=2 过滤兜底。
        parts.append(" ".join(f"dim_{str(k).lower()}={v}" for k, v in sorted(shape.items())))
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
    first_round = bottleneck is None
    if first_round:
        # 首轮无瓶颈画像，BM25 无真实检索信号（P0-2）：置 0，改按 confidence + 策略先验排序。
        bm25_scores = [0.0] * len(scored)
        max_bm25 = 0.0
    else:
        query_terms = _tokenize(_build_query_text(operator_spec, bottleneck, source_features))
        bm25_scores = _bm25(query_terms, [_tokenize(_card_doc_text(c)) for c in scored])
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    hits = []
    for card, bm in zip(scored, bm25_scores):
        bm25_norm = (bm / max_bm25) if max_bm25 > 0 else 0.0
        conf = float(card.get("confidence", 0.5))
        rec = _card_recency(card)
        breakdown = {"bm25_norm": bm25_norm, "confidence": conf, "recency": rec}
        if first_round:
            prior = STRATEGY_PRIOR.get((card.get("transform") or {}).get("name", ""), 0.5)
            breakdown["prior"] = prior
            score = W_PRIOR * prior + W_CONFIDENCE * conf + W_RECENCY * rec
        else:
            score = W_BM25 * bm25_norm + W_CONFIDENCE * conf + W_RECENCY * rec
        hits.append({"card": card, "score": score, "score_breakdown": breakdown})
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


def clamp_confidence(value: float) -> float:
    """把置信度夹在 [CONFIDENCE_MIN, CONFIDENCE_MAX]。"""
    return max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, value))


def _generalize_dim(value) -> str | None:
    """把具体维度值泛化为下界约束字符串（候选卡 semantics.shape 用）。

    规则：向下取 2 的幂（floor pow2），再退一档作下界（1024 → '>=512'，256 → '>=128'）。
    候选卡从单次实验蒸馏而来，写成精确相等（"M": 1024）只会匹配同一尺寸；退一档给出
    可覆盖相邻尺寸的下限，是否过宽交由人工 review 确认。
    """
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v <= 1:
        return ">=1"
    p = 1
    while p * 2 <= v:
        p *= 2
    return f">={max(1, p // 2)}"


def generalize_shape(shape: dict) -> dict:
    """把 shape 每个维度从具体值泛化为下界约束（供候选卡 semantics.shape）。"""
    out = {}
    for k, v in (shape or {}).items():
        g = _generalize_dim(v)
        if g is not None:
            out[k] = g
    return out


def infer_applied_cards(retrieved_ids, prev_feats, cur_feats, cards) -> list[str]:
    """由前后两轮源码特征变化推断模型实际采用了哪张检索到的卡；无法判断的跳过。

    逐张检索到的卡，按其 transform.name 查 CARD_FEATURE_SIGNATURES：
    - tuple(feature, prev_expected, cur_expected)：特征从 prev_expected 翻转到 cur_expected 才判 applied；
    - None 且 name=='register_tiling'：当前 has_shared_mem 与 has_vectorized_load 同时为 True 才判 applied；
    - None 且其它（double_buffering/split_k/未知）：静态特征无法可靠判断，跳过。

    prev_feats/cur_feats 为 roofline.source_features 输出；缺键按 False 处理。
    """
    prev_feats = prev_feats or {}
    cur_feats = cur_feats or {}
    by_id = {c.get("id"): c for c in (cards or [])}
    applied = []
    for cid in retrieved_ids or []:
        card = by_id.get(cid)
        if card is None:
            continue
        name = (card.get("transform") or {}).get("name", "")
        sig = CARD_FEATURE_SIGNATURES.get(name)
        if sig is None:
            if name == "register_tiling" and bool(cur_feats.get("has_shared_mem")) \
                    and bool(cur_feats.get("has_vectorized_load")):
                applied.append(cid)
            continue
        feature, prev_expected, cur_expected = sig
        if bool(prev_feats.get(feature)) == prev_expected and bool(cur_feats.get(feature)) == cur_expected:
            applied.append(cid)
    return applied


def compute_confidence_updates(applied_ids, improved: bool) -> dict[str, float]:
    """每张应用卡本次的置信度增量：improved=True → +0.05，False → -0.05。

    返回「待合并增量」（delta）而非新的绝对置信度；合并时用 clamp_confidence 夹取到
    [CONFIDENCE_MIN, CONFIDENCE_MAX]。knowledge_updates_{N}.json 记录增量，人工 review 后合并。
    """
    delta = CONFIDENCE_STEP if improved else -CONFIDENCE_STEP
    return {cid: delta for cid in (applied_ids or [])}


def _candidate_card_id(run_record: dict) -> str:
    """候选卡 id：SELF_{run_id_clean}-{N}-a{attempt}。

    P1-2：只含 run_id 会让同 run 多尺寸/多轮的候选卡重名，R3c 合并按 id 索引时
    互覆盖。加入 matrix_size 与 attempt 后唯一；文件名与 id 一致。
    """
    rid = str(run_record.get("run_id") or "")
    clean = re.sub(r"[^A-Za-z0-9]", "", rid)
    base = ("SELF_" + clean[:12]) if clean else "SELF_UNKNOWN"
    n = str(run_record.get("matrix_size") or "")
    attempt = str(run_record.get("attempt") or "")
    parts = [base]
    if n:
        parts.append(n)
    if attempt:
        parts.append(f"a{attempt}")
    return "-".join(parts)


def distill_from_run(run_record: dict) -> dict | None:
    """从一轮「成功且更快」的评测结果蒸馏一张候选卡（status='candidate'）。

    run_record 需含：ok、bottleneck、source_features、speedup、run_id、operator_spec。
    - ok=True 且 speedup>1 → 返回候选卡 dict（fingerprint 取自本轮瓶颈/源码特征，
      evidence 记录 run_id/speedup，confidence 初值 0.3）。
    - 其余（失败/未更快）→ None。失败/变慢的 failure_modes 更新需对比前后代码的
      LLM 总结（R4 阶段），此处不做，避免无依据地污染负知识。
    """
    if not run_record.get("ok"):
        return None
    speedup = run_record.get("speedup")
    if speedup is None or speedup <= 1.0:
        return None
    op = run_record.get("operator_spec") or {}
    bottleneck = run_record.get("bottleneck")
    feats = run_record.get("source_features") or {}
    raw_shape = op.get("shape") or {}
    shape = generalize_shape(raw_shape)
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
            "N": raw_shape.get("N"),
        }],
        "confidence": 0.3,
        "provenance": "自身实验蒸馏（self-distillation）",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "last_validated": None,
    }


# ---- R3c：把「待合并增量/证据」真正写回知识库的 CLI（P1-4） ----

def load_updates(results_dir: str) -> dict[str, dict]:
    """扫描 ``results/*/knowledge_updates_*.json``，聚合每张卡的待合并增量与证据。

    返回 ``{card_id: {"delta": float, "evidence": [dict], "sources": [run_id]}}``。
    并发安全（P0-1）：更新文件名已带尺寸，同 run 多尺寸各一份，不会互相覆盖。
    """
    pattern = os.path.join(results_dir, "*", "knowledge_updates_*.json")
    agg: dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        run_id = data.get("run_id")
        for cid, upd in (data.get("updates") or {}).items():
            entry = agg.setdefault(cid, {"delta": 0.0, "evidence": [], "sources": []})
            entry["delta"] = round(entry["delta"] + float(upd.get("delta", 0.0)), 6)
            for ev in upd.get("evidence") or []:
                if run_id and not ev.get("run_id"):
                    ev = dict(ev, run_id=run_id)
                entry["evidence"].append(ev)
            if run_id and run_id not in entry["sources"]:
                entry["sources"].append(run_id)
    return agg


def _evidence_key(ev: dict) -> str:
    """证据去重键：按内容排序后的 JSON（等价证据不重复写入）。"""
    return json.dumps(ev, ensure_ascii=False, sort_keys=True, default=str)


def missing_card_ids(updates: dict[str, dict], strategies_dir: str) -> list[str]:
    """返回 updates 中在 strategies_dir 里没有对应卡文件的 id（merge 时需保留其更新文件）。"""
    return [cid for cid in updates if not os.path.exists(os.path.join(strategies_dir, f"{cid}.json"))]


def _update_file_has_missing(path: str, missing: set[str]) -> bool:
    """判断单个 knowledge_updates_*.json 是否引用了任一缺失的卡 id。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return bool(missing & set((data.get("updates") or {}).keys()))


def consume_update_files(results_dir: str, missing_ids) -> list[str]:
    """消费已合并的 knowledge_updates_*.json，只保留「缺失卡」的增量。

    P1 修复：此前含缺失卡的文件整份保留，导致同一文件里**已合并**卡的 delta 在下次
    merge 时被二次叠加。这里把已合并卡的增量从文件中剔除，只留下缺失卡的增量再写回；
    没有剩余增量的文件才删除。损坏的更新文件保留不删（宁可人工检查，不静默丢数据）。
    返回仍被保留的更新文件路径（下次 merge 继续处理）。
    """
    mset = {str(c) for c in (missing_ids or [])}
    kept: list[str] = []
    pattern = os.path.join(results_dir, "*", "knowledge_updates_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            kept.append(path)
            continue
        remain = {
            cid: upd for cid, upd in (data.get("updates") or {}).items()
            if cid in mset
        }
        if not remain:
            os.remove(path)
            continue
        data["updates"] = remain
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            kept.append(path)
        except OSError:
            kept.append(path)
    return kept


def apply_updates(strategies_dir: str, updates: dict[str, dict], dry_run: bool = False) -> list[str]:
    """把聚合后的置信度增量与证据写回策略卡；返回被修改的卡 id 列表。

    置信度用 clamp_confidence 夹取到 [CONFIDENCE_MIN, CONFIDENCE_MAX]；证据追加到
    卡片 ``evidence`` 列表并按内容去重（P1：反复 merge 不重复堆积）。写卡前先备份为
    ``{cid}.json.bak``（P1：merge 可回滚）。dry_run=True 只报告不落盘。
    """
    changed = []
    for cid, upd in updates.items():
        path = os.path.join(strategies_dir, f"{cid}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            card = json.load(f)
        cur = float(card.get("confidence", 0.5))
        new_conf = clamp_confidence(cur + upd["delta"])
        if not dry_run:
            shutil.copyfile(path, path + ".bak")
            card["confidence"] = new_conf
            existing = card.setdefault("evidence", [])
            seen = {_evidence_key(e) for e in existing}
            for ev in upd.get("evidence") or []:
                k = _evidence_key(ev)
                if k not in seen:
                    existing.append(ev)
                    seen.add(k)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(card, f, ensure_ascii=False, indent=2)
        changed.append(cid)
    return changed


def promote_candidates(candidates_dir: str, strategies_dir: str, dry_run: bool = False) -> list[str]:
    """把候选卡晋升进 strategies/（status 置 'curated'，人工 review 后执行）。

    写成功后删除候选文件，避免二次晋升；dry_run=True 只报告不落盘。
    """
    promoted = []
    for path in sorted(glob.glob(os.path.join(candidates_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                card = json.load(f)
        except Exception:
            continue
        cid = card.get("id")
        if not cid:
            continue
        card["status"] = "curated"
        dest = os.path.join(strategies_dir, f"{cid}.json")
        if not dry_run:
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(card, f, ensure_ascii=False, indent=2)
            os.remove(path)
        promoted.append(cid)
    return promoted


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="RAG 知识库维护 CLI：查看/合并实验增量、晋升候选卡。"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_updates = sub.add_parser("updates", help="查看待合并的置信度增量与证据")
    p_updates.add_argument("--results", default="results")

    p_merge = sub.add_parser("merge", help="把增量/证据写回策略卡并消费更新文件")
    p_merge.add_argument("--results", default="results")
    p_merge.add_argument("--strategies", default=DEFAULT_STRATEGY_DIR)
    p_merge.add_argument("--dry-run", action="store_true", help="只报告不落盘")

    p_promote = sub.add_parser("promote", help="把候选卡晋升进 strategies/")
    p_promote.add_argument("--candidates", default=os.path.join(os.path.dirname(DEFAULT_STRATEGY_DIR), "candidates"))
    p_promote.add_argument("--strategies", default=DEFAULT_STRATEGY_DIR)
    p_promote.add_argument("--dry-run", action="store_true", help="只报告不落盘")

    args = parser.parse_args()
    if args.cmd == "updates":
        agg = load_updates(args.results)
        if not agg:
            print("没有待合并的知识增量。")
            return 0
        for cid, entry in sorted(agg.items()):
            print(f"- {cid}: delta={entry['delta']:+.3f} sources={entry['sources']} "
                  f"evidence={len(entry['evidence'])}条")
        return 0

    if args.cmd == "merge":
        agg = load_updates(args.results)
        changed = apply_updates(args.strategies, agg, dry_run=args.dry_run)
        for cid in changed:
            print(f"{'[dry-run] ' if args.dry_run else ''}合并 {cid}: delta={agg[cid]['delta']:+.3f}")
        missing = missing_card_ids(agg, args.strategies)
        if missing:
            print(f"⚠️ 以下 {len(missing)} 张卡不存在，增量未合并、其更新文件已保留：{sorted(missing)}")
        # 消费已合并的更新文件（非 dry-run）：只保留缺失卡的增量，已合并卡的
        # 增量从文件中剔除，避免下次 merge 对同一张卡二次叠加 delta。
        if not args.dry_run:
            kept = consume_update_files(args.results, missing)
            if kept:
                print(f"保留 {len(kept)} 个含缺失卡的更新文件，待缺失卡补齐后再合并。")
        return 0

    if args.cmd == "promote":
        promoted = promote_candidates(args.candidates, args.strategies, dry_run=args.dry_run)
        for cid in promoted:
            print(f"{'[dry-run] ' if args.dry_run else ''}晋升候选卡 {cid}")
        if not promoted:
            print("没有可晋升的候选卡。")
        return 0

    parser.error(f"未知子命令：{args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
