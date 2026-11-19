"""Web 可视化看板（零新依赖，纯标准库 http.server + 内嵌 HTML/JS）。

对应 ROADMAP 阶段一第 3 条：脱离黑框框，网页端动态展示
「轮次→可信耗时」折线图、相对首轮加速比对比、瓶颈分类与 ncu 硬件指标。

用法：
    python dashboard.py              # 默认 http://127.0.0.1:8000
    python dashboard.py --port 9000  # 指定端口

数据来源：results/index.jsonl + results/{run_id}/{size}.json（async_eval.py 落盘的
results schema v2）。结果目录为空时看板显示「暂无评测数据」，不报错。
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RESULTS_DIR = "results"
DEFAULT_PORT = 8000

# ---- 指标字段名（profiler.py / ncu CSV 解析固定输出）----
_M_SM = "sm__throughput.avg.pct_of_peak_sustained_elapsed"      # SM 计算占用 %
_M_DRAM = "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"  # 显存带宽 %
_M_OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"    # 实际占用率 %
_M_L2 = "lts__t_sector_hit_rate.pct"                            # L2 命中率 %


def _num(v):
    """NaN/None -> None，其余转 float（json 可能带 NaN，看板需安全渲染）。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _clean_record(r: dict) -> dict:
    m = r.get("metrics") or {}
    return {
        "attempt": r.get("attempt"),
        "ok": bool(r.get("ok")),
        "time_sec": _num(r.get("time_sec")),                 # 自报墙钟
        "ncu_time_sec": _num(r.get("ncu_time_sec")),         # 可信 kernel 计时
        "profile_source": r.get("profile_source"),
        "bottleneck": r.get("bottleneck"),
        "sm": _num(m.get(_M_SM)),
        "dram": _num(m.get(_M_DRAM)),
        "occ": _num(m.get(_M_OCC)),
        "l2": _num(m.get(_M_L2)),
        "speedup_vs_prev": _num(r.get("speedup_vs_prev")),
        "model": r.get("model"),
        "tokens": (r.get("usage") or {}).get("total_tokens"),
        "error_feedback": r.get("error_feedback"),
        "source_features": r.get("source_features"),
        "applied_cards": r.get("applied_cards") or [],
        "retrieved_card_ids": r.get("retrieved_card_ids") or [],
    }


def _size_entry_from_full(full: dict, size) -> dict:
    """从某个尺寸的完整落盘 json 构造看板 size_entry（含知识库与计时口径）。"""
    best = (full or {}).get("best") or {}
    best_ncu = _num(best.get("ncu_time_sec"))
    timing_source = "ncu" if best_ncu is not None else None
    if best_ncu is None:
        best_ncu = _num(best.get("effective_time"))
        if best_ncu is not None:
            timing_source = "self"
    knowledge = (full or {}).get("knowledge") or {}
    return {
        "matrix_size": size,
        "model": (full or {}).get("model"),
        "prompt_version": (full or {}).get("prompt_version"),
        "success": bool((full or {}).get("success")),
        "best_attempt": best.get("attempt"),
        "best_time_sec": best_ncu,
        "timing_source": timing_source,
        "best_bottleneck": best.get("bottleneck"),
        "profile_source": best.get("profile_source"),
        "knowledge_enabled": bool(knowledge.get("enabled")),
        "knowledge": {
            "cards_loaded": knowledge.get("cards_loaded"),
            "retrieved_card_ids": knowledge.get("retrieved_card_ids") or [],
            "candidates_created": knowledge.get("candidates_created") or [],
            "knowledge_updates_applied": knowledge.get("knowledge_updates_applied") or {},
        },
        "attempts": [_clean_record(r) for r in (full or {}).get("attempts", [])],
    }


def _load_full(results_dir: str, run_id: str, size) -> dict | None:
    full_path = os.path.join(results_dir, str(run_id), f"{size}.json")
    if not os.path.isfile(full_path):
        return None
    try:
        with open(full_path, encoding="utf-8") as g:
            return json.load(g)
    except (json.JSONDecodeError, OSError):
        return None


def _scan_run_files(results_dir: str) -> list[tuple[str, int, str]]:
    """index.jsonl 缺失时的兜底：扫描 results/*/*.json，目录名当 run_id、文件名当尺寸。"""
    rows: list[tuple[str, int, str]] = []
    if not os.path.isdir(results_dir):
        return rows
    for run_id in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, run_id)
        if not os.path.isdir(run_dir):
            continue
        for fn in sorted(os.listdir(run_dir)):
            if not fn.endswith(".json"):
                continue
            stem = fn[:-5]
            try:
                size = int(stem)
            except ValueError:
                continue
            rows.append((run_id, size, os.path.join(run_dir, fn)))
    return rows


def collect_results(results_dir: str = RESULTS_DIR) -> dict:
    """把 results/ 全部落盘聚合为看板所需结构（纯函数，便于离线单测）。

    返回：
      {
        "env": {…} 或 None,
        "generated_at": iso,
        "runs": [ {run_id, timestamp, sizes: [size_entry, ...]}, ... ]  # 按时间升序
      }
    """
    index_path = os.path.join(results_dir, "index.jsonl")
    runs_map: dict[str, dict] = {}
    env = None
    seen: set[tuple] = set()

    def add(run_id: str, size, timestamp, full):
        nonlocal env
        if not run_id or size is None:
            return
        try:
            size = int(size)
        except (TypeError, ValueError):
            return
        key = (run_id, size)
        if key in seen:
            return
        seen.add(key)
        entry = runs_map.setdefault(
            run_id,
            {"run_id": run_id, "timestamp": timestamp, "sizes": []},
        )
        entry["sizes"].append(_size_entry_from_full(full, size))
        if env is None and (full or {}).get("env"):
            env = full["env"]
        if not entry["timestamp"]:
            entry["timestamp"] = (full or {}).get("timestamp")

    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                run_id = str(row.get("run_id") or "")
                if not run_id or row.get("matrix_size") is None:
                    continue
                add(run_id, row.get("matrix_size"), row.get("timestamp"),
                    _load_full(results_dir, run_id, row.get("matrix_size")))
    else:
        for run_id, size, full_path in _scan_run_files(results_dir):
            full = None
            try:
                with open(full_path, encoding="utf-8") as g:
                    full = json.load(g)
            except (json.JSONDecodeError, OSError):
                full = None
            add(run_id, size, (full or {}).get("timestamp"), full)

    runs = sorted(runs_map.values(), key=lambda r: (r.get("timestamp") or "", r["run_id"]))
    for r in runs:
        r["sizes"].sort(key=lambda s: s["matrix_size"])
    return {"env": env, "generated_at": _now_iso(), "runs": runs}


def _now_iso():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Infra 算子调优 · 评测看板</title>
<style>
:root{
  --bg:#f6f7f9; --card:#ffffff; --fg:#1a202c; --muted:#6b7280;
  --border:#e5e7eb; --accent:#2563eb; --ok:#16a34a; --bad:#dc2626;
  --c256:#2563eb; --c512:#16a34a; --c1024:#d97706;
}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;
  background:var(--bg);color:var(--fg)}
header{padding:20px 28px;background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff}
header h1{margin:0 0 4px;font-size:20px}
header .sub{opacity:.85;font-size:13px}
main{padding:20px 28px;max-width:1200px;margin:0 auto}
.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:16px}
select{padding:7px 10px;border:1px solid var(--border);border-radius:8px;background:#fff;font-size:14px}
button{padding:7px 12px;border:1px solid var(--border);border-radius:8px;background:#fff;font-size:14px;cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .k{color:var(--muted);font-size:12px}
.card .v{font-size:22px;font-weight:600;margin-top:4px}
.card .v small{font-size:12px;font-weight:400;color:var(--muted)}
.panel{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px 18px;margin-bottom:18px}
.panel h2{margin:0 0 4px;font-size:15px}
.panel .hint{color:var(--muted);font-size:12px;margin-bottom:10px}
canvas{width:100%;height:auto;display:block}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.pill.ok{background:#dcfce7;color:var(--ok)}
.pill.bad{background:#fee2e2;color:var(--bad)}
.pill.compute{background:#fee2e2;color:#b91c1c}
.pill.memory{background:#dbeafe;color:#1d4ed8}
.pill.latency{background:#f1f5f9;color:#64748b}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.env{font-size:12px;color:var(--muted);display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 18px}
.env b{color:var(--fg);font-weight:600}
.empty{padding:60px;text-align:center;color:var(--muted)}
footer{padding:16px 28px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>AI Infra 算子调优 · 评测看板</h1>
  <div class="sub">矩阵乘法 CUDA 内核自动优化 · Nsight Compute 硬件画像 · 逐轮演进</div>
</header>
<main>
  <div class="toolbar">
    <label for="runSel">评测批次（run）：</label>
    <select id="runSel"></select>
    <label for="cmpSel">跨 run 对比：</label>
    <select id="cmpSel" multiple size="2" style="min-width:200px;max-width:420px;height:auto" title="按住 Ctrl/Cmd 多选 2+ 个 run"></select>
    <button id="btnRefresh" type="button">刷新</button>
    <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="autoRefresh"> 30s 自动刷新</label>
    <button id="btnCsv" type="button">下载 CSV</button>
    <button id="btnPng" type="button">下载图表 PNG</button>
  </div>
  <div id="envBox" class="panel" style="display:none"><h2>环境快照</h2><div class="env"></div></div>
  <div id="cards" class="cards"></div>
  <div id="empty" class="empty" style="display:none">暂无评测数据 —— 先运行 <code>python async_eval.py</code> 生成 results/。</div>

  <div class="panel"><h2>轮次 → 可信耗时（ncu kernel-only，对数轴）</h2>
    <div class="hint">每个矩阵尺寸一条折线；点的颜色表示该轮瓶颈分类。</div>
    <canvas id="timeChart"></canvas>
    <div class="legend" id="timeLegend"></div>
  </div>

  <div class="panel"><h2>相对首轮加速比（speedup vs attempt 1）</h2>
    <div class="hint">以每个尺寸第一轮成功内核的可信耗时为基线（=1.0），越往上越快。</div>
    <canvas id="speedChart"></canvas>
    <div class="legend" id="speedLegend"></div>
  </div>

  <div class="panel"><h2>ncu 硬件画像指标（按轮次）</h2>
    <div class="hint">SM 计算占用 / 显存带宽 / 实际占用率 / L2 命中率，四组柱状图。</div>
    <canvas id="metricChart"></canvas>
    <div class="legend" id="metricLegend"></div>
  </div>

  <div class="panel"><h2>知识库（RAG）注入与置信度</h2>
    <div class="hint">cards_loaded / 检索卡 / 应用卡 / 候选卡 / 置信度更新，按尺寸聚合。</div>
    <div id="knowledgeBox"></div>
  </div>

  <div class="panel" id="comparePanel" style="display:none">
    <h2>跨 run 对比（尺寸 × 条件，中位数 + 误差棒）</h2>
    <div class="hint">同尺寸下不同 run（kb on/off、模型）的耗时对比；误差棒 = min–max；基线 = 首个选中的 run。</div>
    <canvas id="compareChart"></canvas>
    <div class="legend" id="compareLegend"></div>
    <div id="compareTable" style="margin-top:12px"></div>
  </div>

  <div class="panel"><h2>逐轮明细</h2><div id="tables"></div></div>
</main>
<footer>数据源：results/index.jsonl + results/{run_id}/{size}.json（schema v2） · 零外部依赖 · 由 dashboard.py 生成</footer>

<script>
const SIZECOLORS={256:"#2563eb",512:"#16a34a",1024:"#d97706",2048:"#9333ea"};
const SIZE_PALETTE=["#2563eb","#16a34a","#d97706","#9333ea","#0891b2","#dc2626","#ca8a04","#db2777"];
const BTCOLORS={compute:"#dc2626",memory:"#2563eb",latency:"#94a3b8"};
const BTLABEL={compute:"计算",memory:"访存",latency:"延迟/低占用"};
let DATA=null, RUN=null;

function sizeColor(n){if(SIZECOLORS[n])return SIZECOLORS[n];let h=0;for(const c of String(n))h=(h*31+c.charCodeAt(0))>>>0;return SIZE_PALETTE[h%SIZE_PALETTE.length];}
function runColor(i){return SIZE_PALETTE[i%SIZE_PALETTE.length];}
function median(a){a=[...a].sort((x,y)=>x-y);const m=a.length>>1;return a.length%2?a[m]:(a[m-1]+a[m])/2;}

function fmtTime(s){
  if(s==null) return "—";
  if(s>=1e-3) return (s*1e3).toFixed(2)+" ms";
  if(s>=1e-6) return (s*1e6).toFixed(1)+" µs";
  return (s*1e9).toFixed(0)+" ns";
}
function esc(x){return String(x==null?"":x).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function effTime(a){return (a.ncu_time_sec!=null&&a.ncu_time_sec>0)?a.ncu_time_sec:((a.time_sec!=null&&a.time_sec>0)?a.time_sec:null);}
function isSelfReported(a){return !(a.ncu_time_sec!=null&&a.ncu_time_sec>0)&&(a.time_sec!=null&&a.time_sec>0);}
function firstSuccess(atts){for(const a of atts){if(a.ok&&effTime(a)!=null) return a;}return null;}
function chips(list){if(!list||!list.length)return "—";return list.map(id=>`<span class="pill latency" title="${esc(id)}">${esc(id)}</span>`).join(" ");}

async function load(){
  const r=await fetch("/api/data");
  DATA=await r.json();
  const sel=document.getElementById("runSel");
  sel.innerHTML="";
  if(!DATA.runs.length){
    document.getElementById("empty").style.display="block";
    return;
  }
  document.getElementById("empty").style.display="none";
  const cmp=document.getElementById("cmpSel");
  const cmpPrev=[...cmp.selectedOptions].map(o=>o.value);
  cmp.innerHTML="";
  DATA.runs.forEach((run,i)=>{
    const sizes=run.sizes.map(s=>s.matrix_size).join("/");
    const kb=run.sizes.some(s=>s.knowledge_enabled)?"kb=on":"kb=off";
    const o=document.createElement("option");
    o.value=i;
    o.textContent=`${run.run_id} · ${sizes} · ${kb} · ${(run.timestamp||"").replace("T"," ").slice(0,19)}`;
    sel.appendChild(o);
    cmp.appendChild(o.cloneNode(true));
  });
  [...cmp.options].forEach(o=>{if(cmpPrev.includes(o.value))o.selected=true;});
  sel.onchange=()=>{RUN=DATA.runs[+sel.value];render();};
  cmp.onchange=()=>{renderCompare();};
  sel.value=DATA.runs.length-1;
  RUN=DATA.runs[DATA.runs.length-1];
  render();
  renderCompare();
}

function render(){
  renderEnv(); renderCards();
  const sizes=RUN.sizes;
  const maxAtt=Math.max(1,...sizes.map(s=>Math.max(0,...s.attempts.map(a=>a.attempt||0))));
  drawTimeChart(sizes,maxAtt);
  drawSpeedChart(sizes,maxAtt);
  drawMetricChart(sizes,maxAtt);
  renderKnowledge(sizes);
  renderTables(sizes);
}

function renderEnv(){
  const box=document.getElementById("envBox");
  if(!DATA.env){box.style.display="none";return;}
  box.style.display="block";
  const e=DATA.env, rows=[];
  const kv={GPU: e.gpu, "Nsight Compute": e.ncu_version, "CUDA 镜像": e.cuda_image,
             Python: e.python, numpy: e.numpy, openai: e.openai};
  for(const k in kv){if(kv[k]) rows.push(`<div><b>${k}</b>：${esc(kv[k])}</div>`);}
  box.querySelector(".env").innerHTML=rows.join("");
}

function renderCards(){
  const sizes=RUN.sizes;
  const nOk=sizes.filter(s=>s.success).length;
  const okBest=sizes.filter(s=>s.best_time_sec!=null);
  const bestEntry=okBest.length?okBest.reduce((a,b)=>a.best_time_sec<=b.best_time_sec?a:b):null;
  const bestMin=bestEntry?bestEntry.best_time_sec:null;
  const bestSrc=bestEntry?bestEntry.timing_source:null;
  const totalAtt=sizes.reduce((n,s)=>n+s.attempts.length,0);
  const kbOn=sizes.filter(s=>s.knowledge_enabled).length;
  document.getElementById("cards").innerHTML=[
    card("矩阵尺寸", sizes.map(s=>s.matrix_size).join(" / ")),
    card("成功率", `${nOk}<small>/${sizes.length}</small>`),
    card("总轮次", String(totalAtt)),
    card("最佳可信耗时", fmtTime(bestMin)+(bestSrc==="self"?'<small>自报</small>':'')),
    card("知识库注入", `${kbOn}<small>/${sizes.length} 开</small>`),
  ].join("");
}
function card(k,v){return `<div class="card"><div class="k">${esc(k)}</div><div class="v">${v}</div></div>`;}

function prepCanvas(id){
  const c=document.getElementById(id), dpr=window.devicePixelRatio||1;
  const w=c.parentElement.clientWidth-2;
  c.width=w*dpr; c.height=300*dpr; c.style.height="300px";
  const ctx=c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx,w,h:300};
}

function niceTicks(lo,hi,target=5){
  if(hi===lo) hi=lo+1;
  const step=(hi-lo)/target, mag=Math.pow(10,Math.floor(Math.log10(step)));
  const norm=step/mag, f=norm>=5?5:norm>=2?2:1;
  const s=f*mag, start=Math.ceil(lo/s)*s, out=[];
  for(let v=start;v<=hi+1e-9;v+=s) out.push(v);
  return out;
}

function drawTimeChart(sizes,maxAtt){
  const {ctx,w,h}=prepCanvas("timeChart");
  const pad={l:64,r:14,t:10,b:28};
  const xmin=1,xmax=Math.max(2,maxAtt);
  // y 对数轴：取所有可信耗时（ncu 优先，缺则自报）的最小/最大，向下取整到 10 的幂
  let ys=[];
  sizes.forEach(s=>s.attempts.forEach(a=>{const t=effTime(a);if(t!=null)ys.push(t);}));
  const ylo=ys.length?Math.floor(Math.log10(Math.min(...ys))):-5;
  let yhi=ys.length?Math.ceil(Math.log10(Math.max(...ys))):-3;
  if(yhi<=ylo) yhi=ylo+1;
  const px=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r);
  const py=y=>pad.t+(yhi-Math.log10(y))/(yhi-ylo)*(h-pad.t-pad.b);

  // 网格 + y 刻度
  ctx.strokeStyle="#e5e7eb"; ctx.fillStyle="#6b7280";
  ctx.font="11px sans-serif"; ctx.textAlign="right"; ctx.textBaseline="middle";
  for(let e=ylo;e<=yhi;e++){
    const y=py(Math.pow(10,e));
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText("1e"+e+" s", pad.l-6, y);
  }
  // x 刻度
  ctx.textAlign="center"; ctx.textBaseline="top";
  for(let i=1;i<=xmax;i++){ctx.fillText(String(i),px(i),h-pad.b+6);}
  // y 轴标题
  ctx.save();ctx.translate(14,h/2);ctx.rotate(-Math.PI/2);
  ctx.textAlign="center";ctx.fillText("可信耗时 (s, log)",0,0);ctx.restore();

  // 折线（按尺寸）+ 瓶颈色点；ncu 缺失轮用自报耗时画空心点 + 虚线
  const leg=[];
  sizes.forEach(s=>{
    const pts=[];
    s.attempts.forEach(a=>{const t=effTime(a);if(t!=null&&a.attempt!=null)pts.push({x:a.attempt,y:t,b:a.bottleneck,self:isSelfReported(a)});});
    const col=sizeColor(s.matrix_size);
    leg.push({c:col,l:`${s.matrix_size}×${s.matrix_size}`});
    ctx.lineWidth=2;
    for(let i=0;i<pts.length;i++){
      const p=pts[i];
      if(i>0){
        const q=pts[i-1];
        ctx.beginPath();ctx.setLineDash(p.self||q.self?[5,4]:[]);
        ctx.strokeStyle=col;
        ctx.moveTo(px(q.x),py(q.y));ctx.lineTo(px(p.x),py(p.y));ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    pts.forEach(p=>{
      const X=px(p.x),Y=py(p.y),bc=BTCOLORS[p.b]||"#94a3b8";
      if(p.self){
        ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(X,Y,4.5,0,7);ctx.fill();
        ctx.strokeStyle=bc;ctx.lineWidth=1.5;ctx.stroke();
      }else{
        ctx.fillStyle=bc;ctx.beginPath();ctx.arc(X,Y,4.5,0,7);ctx.fill();
        ctx.strokeStyle="#fff";ctx.lineWidth=1.5;ctx.stroke();
      }
    });
  });
  document.getElementById("timeLegend").innerHTML=leg.map(x=>`<span><i style="background:${x.c}"></i>${x.l}</span>`).join("")
    + `<span><i style="background:#94a3b8"></i>点色=瓶颈：</span>`
    + Object.keys(BTCOLORS).map(k=>`<span><i style="background:${BTCOLORS[k]}"></i>${BTLABEL[k]}</span>`).join("")
    + `<span><i style="background:#fff;border:1px solid #94a3b8"></i>空心点=自报耗时（ncu 缺失）</span>`;
}

function drawSpeedChart(sizes,maxAtt){
  const {ctx,w,h}=prepCanvas("speedChart");
  const pad={l:48,r:14,t:10,b:28};
  const xmin=1,xmax=Math.max(2,maxAtt);
  let ymax=1;
  sizes.forEach(s=>{
    const baseA=firstSuccess(s.attempts);
    if(!baseA) return;
    const baseT=effTime(baseA);
    s.attempts.forEach(a=>{if(a.ok){const t=effTime(a);if(t)ymax=Math.max(ymax,baseT/t);}});
  });
  ymax=Math.ceil(ymax*1.15*10)/10;
  const px=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r);
  const py=y=>pad.t+(ymax-y)/ymax*(h-pad.t-pad.b);

  ctx.strokeStyle="#e5e7eb"; ctx.fillStyle="#6b7280";
  ctx.font="11px sans-serif"; ctx.textAlign="right"; ctx.textBaseline="middle";
  niceTicks(0,ymax).forEach(t=>{
    const y=py(t);
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText(t.toFixed(1)+"×",pad.l-6,y);
  });
  ctx.textAlign="center";ctx.textBaseline="top";
  for(let i=1;i<=xmax;i++)ctx.fillText(String(i),px(i),h-pad.b+6);

  const leg=[];
  sizes.forEach(s=>{
    const baseA=firstSuccess(s.attempts);
    if(!baseA) return;
    const baseT=effTime(baseA);
    const col=sizeColor(s.matrix_size);
    leg.push({c:col,l:`${s.matrix_size}×${s.matrix_size}`});
    const pts=[];
    s.attempts.forEach(a=>{if(a.ok){const t=effTime(a);if(t&&a.attempt!=null)pts.push({x:a.attempt,y:baseT/t,self:isSelfReported(a)});}});
    ctx.lineWidth=2;
    for(let i=0;i<pts.length;i++){
      const p=pts[i];
      if(i>0){
        const q=pts[i-1];
        ctx.beginPath();ctx.setLineDash(p.self||q.self?[5,4]:[]);
        ctx.strokeStyle=col;
        ctx.moveTo(px(q.x),py(q.y));ctx.lineTo(px(p.x),py(p.y));ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    pts.forEach(p=>{
      const X=px(p.x),Y=py(p.y);
      if(p.self){ctx.fillStyle="#fff";ctx.beginPath();ctx.arc(X,Y,3.5,0,7);ctx.fill();ctx.strokeStyle=col;ctx.lineWidth=1.5;ctx.stroke();}
      else{ctx.fillStyle=col;ctx.beginPath();ctx.arc(X,Y,3.5,0,7);ctx.fill();}
    });
  });
  document.getElementById("speedLegend").innerHTML=leg.map(x=>`<span><i style="background:${x.c}"></i>${x.l}</span>`).join("");
}

function drawMetricChart(sizes,maxAtt){
  const {ctx,w,h}=prepCanvas("metricChart");
  const pad={l:48,r:14,t:10,b:28};
  const keys=[["sm","SM 计算%","#dc2626"],["dram","显存带宽%","#2563eb"],
              ["occ","占用率%","#d97706"],["l2","L2 命中%","#16a34a"]];
  const groupW=(w-pad.l-pad.r)/Math.max(1,maxAtt);
  const barW=Math.min(16,(groupW*0.72)/keys.length);
  const px=i=>pad.l+groupW*(i-0.5);
  const py=v=>pad.t+(100-v)/100*(h-pad.t-pad.b);

  // 网格 + 刻度
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#e5e7eb";ctx.fillStyle="#6b7280";
  ctx.font="11px sans-serif";ctx.textAlign="right";ctx.textBaseline="middle";
  [0,25,50,75,100].forEach(t=>{
    const y=py(t);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText(t+"%",pad.l-6,y);
  });
  ctx.textAlign="center";ctx.textBaseline="top";
  for(let i=1;i<=maxAtt;i++)ctx.fillText(String(i),px(i),h-pad.b+6);

  // 逐根画柱
  sizes.forEach(s=>{
    s.attempts.forEach(a=>{
      if(a.attempt==null) return;
      const cx=px(a.attempt)-(keys.length*barW)/2;
      keys.forEach((k,ki)=>{
        const v=a[k[0]]; if(v==null) return;
        ctx.fillStyle=k[2];
        const top=py(v), bot=py(0);
        ctx.fillRect(cx+ki*barW,top,barW-2,bot-top);
      });
    });
  });
  document.getElementById("metricLegend").innerHTML=keys.map(k=>`<span><i style="background:${k[2]}"></i>${k[1]}</span>`).join("");
}

function renderTables(sizes){
  const box=document.getElementById("tables");box.innerHTML="";
  sizes.forEach(s=>{
    const base=firstSuccess(s.attempts);
    const baseT=base?effTime(base):null;
    let rows=s.attempts.map(a=>{
      const t=effTime(a);
      const sp=(a.ok&&t&&baseT)?(baseT/t):null;
      const bt=a.bottleneck;
      const self=(t!=null&&isSelfReported(a));
      return `<tr>
        <td class="num">${a.attempt}</td>
        <td>${a.ok?'<span class="pill ok">✓ 正确</span>':'<span class="pill bad">✗ 错误</span>'}</td>
        <td class="num">${fmtTime(a.ncu_time_sec)}</td>
        <td class="num">${fmtTime(a.time_sec)}${self?' <span class="pill latency">自报</span>':''}</td>
        <td class="num">${sp!=null?sp.toFixed(2)+"×":"—"}</td>
        <td>${bt?`<span class="pill ${esc(bt)}">${BTLABEL[bt]||esc(bt)}</span>`:"—"}</td>
        <td>${esc(a.model||"")}</td>
        <td class="num">${a.tokens!=null?a.tokens:"—"}</td>
        <td>${chips(a.applied_cards)}</td>
        <td>${chips(a.retrieved_card_ids)}</td>
      </tr>`;
    }).join("");
    const best=s.best_time_sec;
    box.insertAdjacentHTML("beforeend",`
      <div style="margin-bottom:14px">
        <h3 style="margin:0 0 6px;font-size:14px">${esc(s.matrix_size)}×${esc(s.matrix_size)} 矩阵
          <span style="color:var(--muted);font-weight:400;font-size:12px">
            · 最佳 ${fmtTime(best)}${s.timing_source==="self"?'（自报）':''}（第 ${esc(s.best_attempt)} 轮 · ${esc(s.best_bottleneck||"—")}）
          </span></h3>
        <table>
          <thead><tr>
            <th class="num">轮次</th><th>结果</th><th class="num">可信耗时 (ncu)</th><th class="num">自报耗时</th>
            <th class="num">加速比</th><th>瓶颈</th><th>实际模型</th><th class="num">tokens</th>
            <th>应用卡</th><th>检索卡</th>
          </tr></thead><tbody>${rows}</tbody>
        </table>
      </div>`);
  });
}

function renderKnowledge(sizes){
  const box=document.getElementById("knowledgeBox");
  const rows=sizes.map(s=>{
    const k=s.knowledge||{};
    const applied=new Set();
    s.attempts.forEach(a=>{(a.applied_cards||[]).forEach(c=>applied.add(c));});
    const upd=k.knowledge_updates_applied||{};
    const updTxt=Object.entries(upd).map(([id,u])=>{
      const cur=u&&u.current_confidence!=null?u.current_confidence.toFixed(2):"—";
      const sug=u&&u.suggested_confidence!=null?u.suggested_confidence.toFixed(2):"—";
      return `${esc(id)} ${cur}→${sug}`;
    }).join("；")||"—";
    return `<tr>
      <td class="num">${s.matrix_size}×${s.matrix_size}</td>
      <td>${s.knowledge_enabled?'<span class="pill ok">开</span>':'<span class="pill latency">关</span>'}</td>
      <td class="num">${k.cards_loaded!=null?k.cards_loaded:"—"}</td>
      <td>${chips(k.retrieved_card_ids)}</td>
      <td>${chips([...applied])}</td>
      <td>${chips(k.candidates_created)}</td>
      <td>${updTxt}</td>
    </tr>`;
  }).join("");
  box.innerHTML=`<table><thead><tr>
    <th class="num">尺寸</th><th>知识库</th><th class="num">已加载卡</th><th>检索卡</th><th>应用卡</th><th>候选卡</th><th>置信度更新 (id cur→sug)</th>
  </tr></thead><tbody>${rows}</tbody></table>`;
}

function runLabel(r){
  const kb=r.sizes.some(s=>s.knowledge_enabled)?"kb=on":"kb=off";
  const models=[...new Set(r.sizes.map(s=>s.model).filter(Boolean))];
  return `${r.run_id} · ${kb}${models.length?", "+models.join(","):""}`;
}

function sizeMedian(r,sz){
  const s=r.sizes.find(x=>x.matrix_size===sz);
  if(!s)return null;
  const ts=[];
  s.attempts.forEach(a=>{if(a.ok){const t=effTime(a);if(t)ts.push(t);}});
  return ts.length?median(ts):null;
}

function drawCompareChart(runs){
  const {ctx,w,h}=prepCanvas("compareChart");
  const pad={l:64,r:14,t:10,b:28};
  const sizes=[...new Set(runs.flatMap(r=>r.sizes.map(s=>s.matrix_size)))].sort((a,b)=>a-b);
  if(!sizes.length)return;
  let ys=[];
  const data={};
  sizes.forEach(sz=>{
    data[sz]=runs.map((r,i)=>{
      const s=r.sizes.find(x=>x.matrix_size===sz);
      const ts=[];
      if(s)s.attempts.forEach(a=>{if(a.ok){const t=effTime(a);if(t)ts.push(t);}});
      if(!ts.length)return null;
      const lo=Math.min(...ts),hi=Math.max(...ts),med=median(ts);
      ys.push(lo,hi);
      return {runIdx:i,med,lo,hi};
    });
  });
  const ylo=ys.length?Math.floor(Math.log10(Math.min(...ys))):-5;
  let yhi=ys.length?Math.ceil(Math.log10(Math.max(...ys))):-3;
  if(yhi<=ylo)yhi=ylo+1;
  const groupW=(w-pad.l-pad.r)/sizes.length;
  const barW=Math.min(36,groupW*0.7/runs.length);
  const px=sz=>{const i=sizes.indexOf(sz);return pad.l+groupW*(i+0.5);};
  const py=y=>pad.t+(yhi-Math.log10(y))/(yhi-ylo)*(h-pad.t-pad.b);
  const bottom=py(Math.pow(10,ylo));

  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle="#e5e7eb";ctx.fillStyle="#6b7280";
  ctx.font="11px sans-serif";ctx.textAlign="right";ctx.textBaseline="middle";
  for(let e=ylo;e<=yhi;e++){
    const y=py(Math.pow(10,e));
    ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();
    ctx.fillText("1e"+e+" s",pad.l-6,y);
  }
  ctx.textAlign="center";ctx.textBaseline="top";
  sizes.forEach(sz=>ctx.fillText(`${sz}×${sz}`,px(sz),h-pad.b+6));
  ctx.save();ctx.translate(14,h/2);ctx.rotate(-Math.PI/2);
  ctx.textAlign="center";ctx.fillText("耗时 (s, log)",0,0);ctx.restore();

  sizes.forEach(sz=>{
    const items=(data[sz]||[]).filter(Boolean);
    const cx=px(sz);
    const totalW=items.length*barW;
    const startX=cx-totalW/2;
    items.forEach((it,k)=>{
      const x=startX+k*barW;
      const col=runColor(it.runIdx);
      ctx.strokeStyle="#9ca3af";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(x+barW/2,py(it.lo));ctx.lineTo(x+barW/2,py(it.hi));ctx.stroke();
      ctx.beginPath();ctx.moveTo(x+barW/2-3,py(it.lo));ctx.lineTo(x+barW/2+3,py(it.lo));ctx.stroke();
      ctx.beginPath();ctx.moveTo(x+barW/2-3,py(it.hi));ctx.lineTo(x+barW/2+3,py(it.hi));ctx.stroke();
      const top=py(it.med), ht=bottom-top;
      ctx.fillStyle=col;ctx.globalAlpha=0.85;
      ctx.fillRect(x,top,barW-2,ht);
      ctx.globalAlpha=1;
    });
  });
}

function renderCompareTable(runs){
  const box=document.getElementById("compareTable");
  const sizes=[...new Set(runs.flatMap(r=>r.sizes.map(s=>s.matrix_size)))].sort((a,b)=>a-b);
  const base=runs[0];
  const rows=sizes.map(sz=>{
    const cells=runs.map(r=>{
      const m=sizeMedian(r,sz);
      const bm=base?sizeMedian(base,sz):null;
      const sp=(m&&bm)?(bm/m):null;
      return `<td class="num">${fmtTime(m)}${sp!=null?`<br><small style="color:var(--muted)">${sp.toFixed(2)}×</small>`:""}</td>`;
    }).join("");
    return `<tr><td class="num">${sz}×${sz}</td>${cells}</tr>`;
  }).join("");
  box.innerHTML=`<table><thead><tr><th class="num">尺寸</th>${runs.map((r,i)=>`<th class="num">${esc(runLabel(r))}${i===0?"（基线）":""}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function renderCompare(){
  const sel=document.getElementById("cmpSel");
  const idxs=[...sel.selectedOptions].map(o=>+o.value);
  const runs=idxs.map(i=>DATA&&DATA.runs[i]).filter(Boolean);
  const panel=document.getElementById("comparePanel");
  if(runs.length<2){panel.style.display="none";return;}
  panel.style.display="block";
  drawCompareChart(runs);
  document.getElementById("compareLegend").innerHTML=runs.map((r,i)=>`<span><i style="background:${runColor(i)}"></i>${esc(runLabel(r))}</span>`).join("");
  renderCompareTable(runs);
}

function download(name,blob){
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);a.download=name;a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
}

function exportCSV(){
  if(!RUN)return;
  const rows=[["run_id","matrix_size","attempt","ok","ncu_time_sec","self_time_sec","speedup_vs_first","bottleneck","model","tokens","applied_cards","retrieved_card_ids"]];
  RUN.sizes.forEach(s=>{
    const base=firstSuccess(s.attempts);
    const baseT=base?effTime(base):null;
    s.attempts.forEach(a=>{
      const t=effTime(a);
      rows.push([RUN.run_id,s.matrix_size,a.attempt,a.ok?"1":"0",
        a.ncu_time_sec??"",a.time_sec??"",
        (a.ok&&t&&baseT)?(baseT/t).toFixed(4):"",
        a.bottleneck||"",a.model||"",a.tokens??"",
        (a.applied_cards||[]).join(";"),(a.retrieved_card_ids||[]).join(";")]);
    });
  });
  const csv=rows.map(r=>r.map(v=>{const s=String(v);return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;}).join(",")).join("\n");
  download(`run_${RUN.run_id}.csv`,new Blob(["\ufeff"+csv],{type:"text/csv;charset=utf-8"}));
}

function exportPNG(){
  ["timeChart","speedChart","metricChart","compareChart"].forEach(id=>{
    const c=document.getElementById(id);
    if(!c||!c.width||c.getClientRects().length===0)return;
    c.toBlob(b=>{if(b)download(`${id}.png`,b);});
  });
}

document.getElementById("btnRefresh").onclick=load;
document.getElementById("autoRefresh").onchange=e=>{
  if(e.target.checked){if(!window.__autoTimer)window.__autoTimer=setInterval(load,30000);}
  else if(window.__autoTimer){clearInterval(window.__autoTimer);window.__autoTimer=null;}
};
document.getElementById("btnCsv").onclick=exportCSV;
document.getElementById("btnPng").onclick=exportPNG;
let __resizeTimer=null;
window.addEventListener("resize",()=>{clearTimeout(__resizeTimer);__resizeTimer=setTimeout(()=>{if(RUN){render();renderCompare();}},200);});

load();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    results_dir = RESULTS_DIR  # main() 里可被 --results 覆盖

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/data":
            data = collect_results(self.results_dir)
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt, *args):
        pass  # 静默访问日志


def main():
    ap = argparse.ArgumentParser(description="AI Infra 评测可视化看板")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="监听端口（默认 8000）")
    ap.add_argument("--host", default="127.0.0.1", help="绑定地址（默认 127.0.0.1）")
    ap.add_argument("--results", default=RESULTS_DIR, help="results 目录（默认 results）")
    args = ap.parse_args()
    Handler.results_dir = os.path.abspath(args.results)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板已启动： http://{args.host}:{args.port}  （Ctrl+C 退出）")
    print(f"数据目录： {Handler.results_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
