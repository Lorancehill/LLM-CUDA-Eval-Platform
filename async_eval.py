import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from openai import AsyncOpenAI
from dotenv import load_dotenv

import validate
import profiler
import roofline
import eval_logic
import rag

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# 限制同时最多 1 个任务占用 GPU
gpu_sem = asyncio.Semaphore(1)

# Docker 沙箱运行超时上限（秒）
TIMEOUT_SECONDS = 20
# ncu 画像超时上限（秒）：ncu 需多次 replay kernel 采集全部指标，比普通运行慢得多
NCU_TIMEOUT_SECONDS = 120
# 内核计时重复次数（论文计时口径 3.1.4：跑 N 次取中位数，抗首轮冷启动/偶发抖动）
def _env_int(name: str, default: int) -> int:
    """读取正整数环境变量；缺失/非法/<=0 时回退默认值（坏环境变量不应让整批评测崩溃）。"""
    try:
        v = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


TIMING_REPEATS = _env_int("TIMING_REPEATS", 5)
# API 调用超时 / 重试次数（B6：一次 API 抖动不应毁掉整批）
API_TIMEOUT_SECONDS = 60
API_RETRIES = 3

# 沙箱镜像：必须支持 Blackwell (sm_120)，即 CUDA >= 12.8。
# 12.0.x 不支持 RTX 5060 的 compute_120，编译出的内核无法在该 GPU 上运行。
CUDA_IMAGE = "nvidia/cuda:12.8.2-devel-ubuntu22.04"
# 首次拉取镜像的超时上限（秒）；镜像数 GB，需远大于执行超时
PULL_TIMEOUT_SECONDS = 900
# 拉取重试次数；Docker Hub 走 CDN，弱网下可能偶发 connection timed out，
# docker pull 可断点续传，重试只补缺的层
PULL_RETRIES = 3

# 画像/分类/反馈证据链落盘目录（运行时产物，已在 .gitignore 忽略）
RESULTS_DIR = "results"

# 配置的 API 模型别名；真实服务模型以 response.model 为准，逐轮落盘。
# 可用 MODEL 环境变量覆盖（P0-1：批跑脚本按 run 注入不同模型）。
MODEL_NAME = os.getenv("MODEL", "deepseek-chat") or "deepseek-chat"
PROMPT_VERSION = "matmul_v1"

# 目标内核名 demangled 正则（模型生成的 matmul 内核常见命名）。
# 多 kernel 代码里用 --kernel-name 过滤，避免误采 init/融合算子等第一个 launch。
MATMUL_KERNEL_REGEX = r"(matmul|matMul|MatMul|Matmul|matrixMul|matrix_mul|sgemm|gemm|GEMM)"

# RAG 策略卡目录（RAG 设计 §1：knowledge/strategies/*.json）。
# 用 rag.DEFAULT_STRATEGY_DIR（基于 __file__ 的绝对路径），避免 cwd 依赖（P0-1）。
STRATEGY_DIR = rag.DEFAULT_STRATEGY_DIR

# R3 自蒸馏候选卡落盘目录（基于 STRATEGY_DIR 派生的绝对路径，避免 cwd 依赖；P2-2）。
# 人工 review 后再合并进 strategies/，避免自动污染知识库。
CANDIDATES_DIR = os.path.join(os.path.dirname(rag.DEFAULT_STRATEGY_DIR), "candidates")


def _knowledge_config():
    """读取 RAG 消融开关（设计 §4）。默认关闭，保持与引入 RAG 前行为一致。

    - KNOWLEDGE_ENABLED=1/0：知识库开/关（消融主开关）
    - KNOWLEDGE_TOPK：每次注入的策略卡条数（默认 3）
    - mode 固定 'cards'：MVP 仅实现策略卡检索；docs/hybrid 为升级路径（设计 §3.1），未实现。
    """
    return {
        "enabled": os.getenv("KNOWLEDGE_ENABLED", "0") in ("1", "true", "True", "yes", "on"),
        "top_k": _env_int("KNOWLEDGE_TOPK", 3),
        "mode": "cards",
    }


def build_prompt(matrix_size, knowledge_block=None):
    reqs = (
        f"请用C++和CUDA写一个 {matrix_size}*{matrix_size} 的矩阵乘法代码。要求：\n"
        "1. 矩阵类型用 float（32位浮点），初始化 A[i][j] = ((i+j)%10)*0.1f，B[i][j] = ((i*3+j)%10)*0.1f；\n"
        "2. 用 wall-clock 测量 kernel 执行耗时（用 std::chrono::high_resolution_clock 或 CUDA event，"
        "在 kernel 启动前后各调一次 cudaDeviceSynchronize；不要用 clock()，它测的是 CPU 时间），"
        "并打印出 'Time: xxx seconds'；\n"
        "3. 打印 'C[0][0] = xxx' 用于结果校验；\n"
        f"4. 将结果矩阵 C（先 cudaMemcpy 从 device 拷回 host）以二进制写入文件 'cuda_output_{matrix_size}.bin'："
        f"fopen('cuda_output_{matrix_size}.bin','wb') 后 fwrite(C, sizeof(float), {matrix_size}*{matrix_size}, fp)。\n"
    )
    if knowledge_block:
        # 知识库检索结果放在靠后位置（RAG 设计 §5.4：靠后 + 显式引用要求）
        reqs += "\n" + knowledge_block + "\n"
    reqs += "不要任何废话，只要代码。"
    return reqs


async def _docker_run(cmd, container_name, timeout=TIMEOUT_SECONDS):
    """在 timeout 秒内运行 docker 命令；超时则强杀容器并返回已捕获的部分输出。

    返回 (returncode, stdout_bytes, stderr_bytes, timed_out)：
    - 正常结束：returncode 为进程退出码，timed_out=False；
    - 超时：returncode=None，stdout/stderr 为超时前已捕获的部分输出，timed_out=True。
    """
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd, shell=True, capture_output=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        # --rm 之外再兜底强杀一次，避免残留容器占用 GPU
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
        return None, (e.stdout or b""), (e.stderr or b""), True


async def _run_host(cmd_list, timeout):
    """宿主侧子进程（ncu 等）。返回 (returncode, stdout_bytes, stderr_bytes, timed_out)。"""
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd_list, capture_output=True, timeout=timeout
        )
        return proc.returncode, proc.stdout, proc.stderr, False
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or b""), (e.stderr or b""), True


def _usage_dict(usage):
    """把 OpenAI usage 对象转成可落盘的 dict（缺失/异常返回 None）。"""
    if usage is None:
        return None
    try:
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    except Exception:
        return None


async def _chat_completion(messages):
    """带超时/退避重试的 API 调用。返回 (response, usage_dict, error)。

    response 为 None 表示 API_RETRIES 次均失败（error 为最后一次异常）。
    """
    last_err = None
    for attempt in range(1, API_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME, messages=messages, timeout=API_TIMEOUT_SECONDS
            )
            return response, _usage_dict(getattr(response, "usage", None)), None
        except Exception as e:
            last_err = e
            print(f"⚠️ API 调用失败（第 {attempt}/{API_RETRIES} 次）：{e}")
            if attempt < API_RETRIES:
                await asyncio.sleep(2 ** (attempt - 1))  # 1, 2, 4s 退避
    return None, None, last_err


async def ensure_image():
    """评测前确保沙箱镜像已拉取。

    docker run 首次遇到未缓存镜像会边跑边拉（数 GB），耗时远超 TIMEOUT_SECONDS，
    导致每次 attempt 都被 20s 超时误杀。这里显式先拉取，把拉取与执行超时解耦。
    返回 True 表示镜像可用，False 表示拉取失败（应中止评测）。
    """
    try:
        r = await asyncio.to_thread(
            subprocess.run,
            ["docker", "image", "inspect", CUDA_IMAGE],
            capture_output=True,
            timeout=30,
        )
        if r.returncode == 0:
            print(f"[镜像] {CUDA_IMAGE} 已缓存，跳过拉取。")
            return True
    except Exception:
        pass

    print(f"[镜像] 拉取 {CUDA_IMAGE}（首次较慢，请耐心等待）...")
    for attempt in range(1, PULL_RETRIES + 1):
        print(f"[镜像] 拉取尝试 {attempt}/{PULL_RETRIES} ...")
        try:
            r = await asyncio.to_thread(
                subprocess.run,
                ["docker", "pull", CUDA_IMAGE],
                capture_output=True,
                timeout=PULL_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(f"[镜像] ⚠️ 拉取超时（>{PULL_TIMEOUT_SECONDS}s）。")
            if attempt < PULL_RETRIES:
                continue
            print(f"[镜像] ❌ 多次超时。请手动执行：docker pull {CUDA_IMAGE}")
            return False
        except Exception as e:
            # docker 不存在 / 守护进程未启动等：优雅返回 False，而不是让异常穿透整个评测
            print(f"[镜像] ⚠️ 拉取异常：{e}")
            if attempt < PULL_RETRIES:
                continue
            return False
        if r.returncode == 0:
            print("[镜像] ✅ 拉取完成。")
            return True
        print(f"[镜像] ⚠️ 拉取失败：{r.stderr.decode(errors='replace')[-300:]}")
        if attempt < PULL_RETRIES:
            print("[镜像] 断点续传重试中...")
            continue
    print(f"[镜像] ❌ 拉取 {PULL_RETRIES} 次均失败。请检查网络后手动执行：docker pull {CUDA_IMAGE}")
    return False


_ncu_bin_cache: str | None = None
_ncu_gb206_cache: bool | None = None


def _ncu_supports_gb206(ncu_bin: str) -> bool:
    """运行 ``ncu --list-chips`` 判断该二进制是否认识 gb206（Blackwell 关键判据）。"""
    try:
        chips = subprocess.run([ncu_bin, "--list-chips"], capture_output=True, timeout=15, text=True)
    except Exception:
        return False
    return "gb206" in ((chips.stdout or "") + (chips.stderr or "")).lower()


def _resolve_ncu() -> tuple[str, bool]:
    """在候选里挑第一个认识 gb206 的 ncu 二进制，返回 (bin, gb206_ok)。

    新版 ncu（2026.x，支持 gb206）常装在 ``~/.local/nsight-compute/*/ncu`` 且不进 PATH，
    系统 PATH 里的 ``/usr/bin/ncu`` 还是 2024.1.1（不认识 gb206）。逐个候选跑
    ``--list-chips``，命中 gb206 才采用；否则退回首选候选（上层会告警并降级 roofline）。
    """
    candidates = profiler.list_ncu_candidates()
    for cand in candidates:
        if _ncu_supports_gb206(cand):
            return cand, True
    return (candidates[0] if candidates else "ncu"), False


def _detect_ncu_once() -> tuple[str, bool]:
    """只探测一次 ncu，get_ncu_bin / ncu_supports_gb206 共享结果。

    否则两个函数各自调用 _resolve_ncu()，会把 --list-chips 子进程跑两遍。
    """
    global _ncu_bin_cache, _ncu_gb206_cache
    if _ncu_bin_cache is None:
        _ncu_bin_cache, _ncu_gb206_cache = _resolve_ncu()
    return _ncu_bin_cache, bool(_ncu_gb206_cache)


def get_ncu_bin() -> str:
    """返回本次运行要用的 ncu 二进制路径（懒加载 + 缓存，全进程只探测一次）。"""
    return _detect_ncu_once()[0]


def ncu_supports_gb206() -> bool:
    """本次运行采用的 ncu 是否认识 gb206（与 get_ncu_bin 共享同一次探测）。"""
    return _detect_ncu_once()[1]


def probe_host_ncu():
    """探测本次运行将采用的 ncu 是否可用且认识 gb206，返回 (ok, message)。"""
    ncu_bin = get_ncu_bin()
    try:
        r = subprocess.run([ncu_bin, "--version"], capture_output=True, timeout=15, text=True)
    except FileNotFoundError:
        return False, f"宿主未安装 ncu（候选路径均不可用：{profiler.list_ncu_candidates()}）"
    except Exception as e:
        return False, f"ncu 探测异常：{e}"
    if r.returncode != 0:
        return False, f"ncu 无法执行：{(r.stderr or r.stdout).strip()[-120:]}"
    if ncu_supports_gb206():
        return True, f"ncu 可用且认识 gb206（{ncu_bin}）"
    return False, f"ncu 已安装但不认识 gb206（--list-chips 无 gb206，需升级；当前用 {ncu_bin}）"


def collect_experiment_meta():
    """从环境变量读取实验元数据（周末批跑的消融标签，P0-1）。

    - EXPERIMENT_ID / CONDITION / REPEAT：由 run_experiment.py 以子进程环境注入；
    - MODEL_NAME：本次运行配置的模型别名（由 MODEL 环境变量解析）。
    未设置时返回 None，便于区分「独立运行」与「批跑条件 run」。
    """
    return {
        "EXPERIMENT_ID": os.getenv("EXPERIMENT_ID"),
        "CONDITION": os.getenv("CONDITION"),
        "REPEAT": os.getenv("REPEAT"),
        "MODEL_NAME": MODEL_NAME,
    }


def _git_snapshot():
    """记录代码版本（论文可复现底线，P0-2）：git_commit=HEAD，git_dirty=工作区是否干净。

    任一步失败只记 None；git_dirty 仅在 git status 成功执行时有意义（False=干净）。
    """
    snap = {"git_commit": None, "git_dirty": None}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, timeout=10, text=True)
        if r.returncode == 0:
            snap["git_commit"] = r.stdout.strip() or None
    except Exception:
        pass
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, timeout=10, text=True)
        if r.returncode == 0:
            snap["git_dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return snap


def collect_env_snapshot():
    """收集环境快照（GPU/驱动/ncu/镜像/依赖版本），供论文复现；任一步失败只记 unavailable。"""
    import platform

    snap = {"python": platform.python_version()}
    try:
        import numpy
        snap["numpy"] = numpy.__version__
    except Exception:
        snap["numpy"] = "unavailable"
    try:
        import openai as _oai
        snap["openai"] = getattr(_oai, "__version__", "unknown")
    except Exception:
        snap["openai"] = "unavailable"

    def _run(args, timeout=10):
        try:
            r = subprocess.run(args, capture_output=True, timeout=timeout, text=True)
            return (r.stdout or r.stderr or "").strip()
        except Exception as e:
            return f"unavailable: {e}"

    snap["gpu"] = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    ) or "unavailable"
    # GPU 温度/功耗/时钟（论文计时口径 3.1.4：记录运行态环境，佐证计时可信度）
    snap["gpu_temp_power"] = _run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"]
    ) or "unavailable"
    ncu_bin = get_ncu_bin()
    snap["ncu_bin"] = ncu_bin
    ncu_ver = _run([ncu_bin, "--version"])
    # 记录含 "Version" 的完整版本行（如 "Version 2026.4.0.0 …"）；第一行只是版权标题。
    snap["ncu_version"] = _ncu_version_line(ncu_ver)
    snap["cuda_image"] = _run(
        ["docker", "image", "inspect", CUDA_IMAGE, "--format", "{{.Id}}"]
    ) or "unavailable"
    # 实验元数据（P0-1）+ 代码版本（P0-2），供消融对比与论文复现
    snap.update(collect_experiment_meta())
    snap.update(_git_snapshot())
    return snap


def _ncu_version_line(ncu_ver: str) -> str:
    """取 ncu --version 里含 'Version' 的那一行（完整版本号）；无则退回首行。"""
    if not ncu_ver or ncu_ver.startswith("unavailable"):
        return ncu_ver or "unavailable"
    for line in ncu_ver.splitlines():
        if "Version" in line:
            return line.strip()
    return ncu_ver.splitlines()[0].strip() if ncu_ver.strip() else "unavailable"


async def _profile_kernel(matrix_size, binary, kernel_name=None):
    """在宿主侧（WSL2）用 ncu profile 容器编译出的二进制；失败/无指标返回 None。

    依赖用户已：① Windows 开 GPU 性能计数器 ② WSL 装支持 GB206 的 ncu。
    注意：容器编译出的二进制在宿主运行需宿主有对应 CUDA runtime（libcudart）；
    若缺失，ncu 会加载失败，此处优雅降级（上层回落到 roofline proxy）。
    """
    cmd_list = shlex.split(profiler.build_ncu_cmd(binary, kernel_name, ncu_bin=get_ncu_bin()))
    async with gpu_sem:
        try:
            _, stdout, stderr, _ = await _run_host(cmd_list, NCU_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"[{matrix_size}] ⚠️ ncu 画像失败（宿主异常）: {e}")
            return None

    stdout_str = stdout.decode(errors="replace")
    stderr_str = stderr.decode(errors="replace")
    metrics = profiler.parse_ncu_csv(stdout_str)
    if not metrics:
        tail = (stderr_str or stdout_str).strip()[-200:]
        print(f"[{matrix_size}] ⚠️ ncu 未抓到硬件指标，跳过画像（降级 roofline proxy）。{tail}")
        return None
    return metrics


def _empty_record(out_bin):
    return {
        "attempt": None,
        "ok": False,
        "time_sec": None,
        "ncu_time_sec": None,
        "profile_source": None,
        "max_diff": None,
        "cu_filename": None,
        "binary": None,
        "out_bin": out_bin,
        "bottleneck": None,
        "metrics": None,
        "feedback": None,
        "code": None,
        "source_features": None,
        "speedup_vs_prev": None,
        "applied_cards": [],
        "retrieved_card_ids": None,
        "error_feedback": None,
        "compile_errors": [],
        "compile_stdout_tail": None,
        "compile_stderr_tail": None,
        "timed_out": None,
        "returncode": None,
        "model": None,
        "usage": None,
    }


def _read_source_context(cu_filename, compile_errors, before=2, after=2):
    """读出错行附近的 .cu 源码片段（带行号），供自我纠错把出错行喂回模型。

    只取第一条带行号的报错行（line 指 .cu 文件内行号）；读不到文件 / 无行号返回 None，
    上层兜底只给报错文本。
    """
    target = None
    for e in compile_errors or []:
        if e.get("line") is not None:
            target = e["line"]
            break
    if target is None:
        return None
    try:
        with open(cu_filename, encoding="utf-8", errors="replace") as f:
            src = f.readlines()
    except OSError:
        return None
    lo = max(1, target - before)
    hi = min(len(src), target + after)
    out = []
    for n in range(lo, hi + 1):
        marker = ">>>" if n == target else "   "
        out.append(f"{marker} {n:4d} | {src[n - 1].rstrip()}")
    return "\n".join(out)


async def evaluate_matrix(matrix_size, run_id, env_snapshot):
    kb = _knowledge_config()
    cards = rag.load_cards(STRATEGY_DIR) if kb["enabled"] else []
    knowledge_warning = None
    if kb["enabled"]:
        if not cards:
            knowledge_warning = (
                f"KNOWLEDGE_ENABLED=1 但未加载到任何策略卡（目录 {STRATEGY_DIR}）；"
                "本次运行退化为 no-KB 组"
            )
            print(f"[知识库] ⚠️ {knowledge_warning}")
        else:
            print(f"[知识库] 已加载 {len(cards)} 张策略卡")
    operator_spec = {
        "operator": "matmul",
        "shape": {"M": matrix_size, "N": matrix_size, "K": matrix_size, "batch": 1},
        "dtype": "float32",
    }
    # 本次 run 检索到的策略卡 id 集合（落盘 results 的 knowledge 字段，供消融回溯）
    retrieved_card_ids = set()

    def retrieve_hits(bottleneck=None, source_features=None, ncu_metrics=None):
        if not kb["enabled"]:
            return []
        return rag.retrieve(operator_spec, bottleneck=bottleneck, source_features=source_features,
                            ncu_metrics=ncu_metrics, top_k=kb["top_k"], cards=cards)

    # 首轮注入（RAG 设计 §3.3）：尚未画像，仅按算子规格检索该算子族的常用优化手段
    first_hits = retrieve_hits()
    knowledge_block = rag.build_context_block(first_hits) if first_hits else None
    for h in first_hits:
        retrieved_card_ids.add(h["card"]["id"])
    messages = [{"role": "user", "content": build_prompt(matrix_size, knowledge_block)}]
    max_retries = 3

    # 参考结果（float64，见 validate.reference_matmul）
    ref = validate.reference_matmul(matrix_size)
    out_bin = f"cuda_output_{matrix_size}.bin"

    # 记录每一次 attempt 的内核耗时（秒）；无输出（超时/编译失败）记 None。
    # timings = 自报 Time（对照用）；ncu_timings = ncu 可信计时（论文口径 3.1.3）。
    timings = []
    ncu_timings = []

    # best-so-far：只在「结果正确 且 更快」时更新。即使后续 attempt 把模型改坏，
    # 也不会丢掉前面已找到的正确且更快的内核（P0：成功后继续优化，失败会「反杀」成功）。
    best = {
        "ok": False,
        "attempt": None,
        "cu_filename": None,
        "binary": None,
        "effective_time": None,   # 比较「更快」用的耗时（优先 ncu 可信计时，其次自报）
        "self_reported_time": None,
        "ncu_time_sec": None,
        "max_diff": None,
        "bottleneck": None,
        "profile_source": None,
        "metrics": None,
        "feedback": None,
    }
    # 每轮结构化记录，落盘用于回溯「瓶颈 → 策略 → 收益」证据链
    records = []

    # R3 归因/自蒸馏的跨轮状态：追踪「上一成功轮」的源码特征/检索卡/有效耗时，
    # 用于推断本轮实际应用了哪张卡（applied_cards）并计算加速比（speedup_vs_prev）。
    prev_feats = None            # 上一成功轮的源码特征（roofline.source_features 输出）
    prev_retrieved = []          # 上一轮检索并注入的卡 id（本轮尝试应用的对象）
    prev_time = None             # 上一成功轮的有效耗时（秒）
    first_ok_time = None         # 第一次成功 attempt 的有效耗时（蒸馏 speedup 基准）
    best_source_features = None  # 当前 best 内核的源码特征（蒸馏候选卡指纹）
    best_applied_cards = []      # 当前 best attempt 实际应用的卡（P1-1：非空则只追加证据，不蒸馏新卡）
    confidence_events = {}       # card_id -> {delta, current_confidence, rounds[], evidence[]}

    cards_by_id = {c.get("id"): c for c in cards}

    def accumulate_confidence(cids, improved, attempt, applied, speedup=None):
        """把一轮的置信度增量累加到 confidence_events（失败轮 improved=False 惩罚上轮注入卡）。

        提速且真正应用时（improved 且 applied）追加 evidence（run_id/matrix_size/speedup），
        供 R3c 合并 CLI 写回卡片 evidence（P1-4）。"""
        for cid in cids:
            card = cards_by_id.get(cid)
            current = float(card.get("confidence", 0.5)) if card else 0.5
            delta = rag.compute_confidence_updates([cid], improved).get(cid, 0.0)
            ev = confidence_events.setdefault(cid, {
                "delta": 0.0, "current_confidence": current, "rounds": [],
            })
            ev["delta"] = round(ev["delta"] + delta, 6)
            ev["rounds"].append({"attempt": attempt, "applied": bool(applied), "improved": bool(improved)})
            if improved and applied and speedup is not None:
                ev.setdefault("evidence", []).append({
                    "run_id": run_id,
                    "matrix_size": matrix_size,
                    "attempt": attempt,
                    "speedup": round(float(speedup), 4),
                })

    for attempt in range(max_retries):
        print(f"\n[{matrix_size}] ==========================================")
        print(f"[{matrix_size}] 🚀 开始第 {attempt + 1} 次生成与评测...")

        response, usage, api_err = await _chat_completion(messages)
        if response is None:
            record = _empty_record(out_bin)
            record.update({
                "attempt": attempt + 1,
                "error_feedback": f"API 调用失败（{api_err}）",
            })
            records.append(record)
            print(f"[{matrix_size}] ❌ 第 {attempt + 1} 次 API 调用失败，跳过本轮。")
            continue

        choices = getattr(response, "choices", None) or []
        if not choices or not getattr(choices[0].message, "content", None):
            record = _empty_record(out_bin)
            record.update({
                "attempt": attempt + 1,
                "error_feedback": "API 返回空内容（choices 为空或 content 为空）",
            })
            records.append(record)
            print(f"[{matrix_size}] ❌ 第 {attempt + 1} 次 API 返回空内容，跳过本轮。")
            continue
        raw_text = choices[0].message.content or ""
        model = getattr(response, "model", MODEL_NAME)

        # 正则清洗：先取第一个 ``` 围栏内的内容，再只剥掉「已知语言标签 + 换行」，
        # 避免把代码首 token（如 __global__）误当语言标签吃掉。
        fence = re.search(r"```(.*?)```", raw_text, re.DOTALL)
        if fence:
            pure_code = fence.group(1).strip("\n")
            pure_code = re.sub(
                r"^[ \t]*(?:cpp|CPP|c\+\+|C\+\+|cuda|CUDA|cu|CU|c|C)[ \t]*\r?\n",
                "", pure_code,
            )
            pure_code = pure_code.strip()
        else:
            pure_code = raw_text

        # 每轮用独立文件名（attempt 后缀）：避免同名覆盖前面正确且更快的内核源码/二进制
        cu_filename = f"matrix_mul_{matrix_size}_attempt{attempt + 1}.cu"
        binary = f"test_{matrix_size}_attempt{attempt + 1}"

        with open(cu_filename, "w") as f:
            f.write(pure_code)
        print(f"[{matrix_size}] ✅ 代码已写入 {cu_filename}，准备送入沙箱测试...")

        # 清理上次的输出，避免读到过期矩阵
        if os.path.exists(out_bin):
            os.remove(out_bin)

        # 这里加了 --rm 自动清理容器，并执行编译和运行
        container_name = f"test_{matrix_size}_attempt{attempt + 1}"
        # 内核计时跑 TIMING_REPEATS 次取中位数（论文口径 3.1.4：抗首轮冷启动/抖动）
        run_repeats = " && ".join(f"./{binary}" for _ in range(TIMING_REPEATS))
        cmd = (
            f"docker run --rm --name {container_name} --memory=2g --cpus=1.0 --network none "
            f"-v $(pwd):/app --gpus all {CUDA_IMAGE} "
            f"bash -c 'cd /app && nvcc -arch=sm_120 {cu_filename} -o {binary} && {run_repeats}'"
        )

        # 异步执行外部命令（GPU 串行，API 并发）
        async with gpu_sem:
            try:
                returncode, stdout, stderr, timed_out = await _docker_run(cmd, container_name)
            except Exception as e:
                # docker 二进制缺失 / daemon 未启动等，也作为错误反馈给 DeepSeek
                print(f"[{matrix_size}] ❌ Docker 启动失败: {e}")
                returncode, stdout, stderr, timed_out = None, b"", str(e).encode(), False

        stdout_str = stdout.decode(errors="replace")
        stderr_str = stderr.decode(errors="replace")

        # 记录本次 attempt 的内核耗时（秒）；无 'Time:' 输出记 None。
        # 论文口径：TIMING_REPEATS 次重复取中位数（自报 Time 仅作对照，见 3.1.3）。
        time_sec = eval_logic.parse_time_seconds_median(stdout_str)
        timings.append(time_sec)
        ncu_timings.append(None)  # 成功轮画像后被覆盖为 ncu 可信计时

        # 读取 CUDA 输出矩阵，与 numpy 参考做按尺寸校准的容差校验
        ok = False
        max_diff = None
        if os.path.exists(out_bin):
            try:
                got = validate.load_matrix(out_bin, matrix_size)
                rtol, atol = validate.default_tolerance(matrix_size)
                ok, max_diff = validate.matrices_close(ref, got, rtol=rtol, atol=atol)
            except ValueError as e:
                print(f"[{matrix_size}] ⚠️ 矩阵读取/校验异常: {e}")

        record = _empty_record(out_bin)
        record.update({
            "attempt": attempt + 1,
            "ok": ok,
            "time_sec": time_sec,
            "max_diff": max_diff,
            "cu_filename": cu_filename,
            "binary": binary,
            "code": pure_code,
            "timed_out": timed_out,
            "returncode": returncode,
            "model": model,
            "usage": usage,
        })

        if not ok:
            print(f"[{matrix_size}] ❌ 第 {attempt + 1} 次评测失败！正在反馈给 AI...")
            messages.append({"role": "assistant", "content": raw_text})

            err_tail = stderr_str[-300:].strip()
            out_tail = stdout_str[-300:].strip()
            record["compile_stdout_tail"] = out_tail
            record["compile_stderr_tail"] = err_tail

            # 真正的自我纠错：优先结构化编译错误（nvcc/g++/ptxas 行号 + 报错原因）
            # + 出错行源码上下文，喂回模型；找不到结构化错误时兜底输出尾部 300 字符。
            compile_errors = eval_logic.extract_compile_errors(stderr_str + "\n" + stdout_str)
            record["compile_errors"] = compile_errors
            source_context = _read_source_context(cu_filename, compile_errors)

            if timed_out:
                kind = "timeout"
            elif returncode is None or returncode != 0:
                kind = "run_failed"
            elif max_diff is None:
                kind = "no_output"
            else:
                kind = "wrong_result"
            error_feedback = eval_logic.build_error_feedback(
                compile_errors, source_context, kind,
                returncode=returncode, max_diff=max_diff, out_bin=out_bin,
                timeout=TIMEOUT_SECONDS, stdout_tail=out_tail, stderr_tail=err_tail,
            )
            record["error_feedback"] = error_feedback
            messages.append({"role": "user", "content": error_feedback})
            # R3：失败轮对上轮注入的卡给负置信度（模型被要求应用但编译/结果失败）
            if prev_retrieved:
                accumulate_confidence(prev_retrieved, False, attempt + 1, applied=False)
            records.append(record)
            continue

        print(f"[{matrix_size}] 🎉 第 {attempt + 1} 次评测成功！结果与基准一致（max diff {max_diff:.3g}）。")
        for line in stdout_str.split('\n'):
            if "Time:" in line or "耗时" in line:
                print(f"[{matrix_size}] --- 跑分: {line.strip()}")

        # 结果正确后画像：优先宿主 ncu，失败降级 roofline proxy（counter-free）
        profile = await _profile_kernel(matrix_size, binary, MATMUL_KERNEL_REGEX)
        label = None
        feedback = None
        ncu_time_sec = None
        source = None
        metrics = None
        if profile is not None:
            source = "ncu"
            metrics = profile
            print(f"[{matrix_size}] 📊 ncu 硬件画像：\n{profiler.format_metrics(profile)}")
            label, feedback = profiler.classify_bottleneck(profile)
            raw_ncu_ns = profile.get("gpu__time_duration.sum")
            # 0/负值视为无效（指标缺失或解析异常），避免 best_time=0 污染保优与加速比
            if raw_ncu_ns is not None and raw_ncu_ns > 0:
                ncu_time_sec = raw_ncu_ns / 1e9
                print(f"[{matrix_size}] ⏱️  ncu 可信计时 {ncu_time_sec:.6f}s（自报 {time_sec if time_sec is not None else 'N/A'}s）")
        else:
            # 降级：roofline proxy（不依赖 ncu，今天即可端到端跑通）
            source = "roofline_proxy"
            label, feedback, metrics = roofline.roofline_classify(matrix_size, time_sec, pure_code)
            print(f"[{matrix_size}] 📊 roofline 指纹（counter-free）：{roofline.format_proxy_metrics(metrics)}")
        if label not in (None, "unknown"):
            print(f"[{matrix_size}] 🧭 瓶颈分类: {label}（{source}）")

        record["ncu_time_sec"] = ncu_time_sec
        ncu_timings[-1] = ncu_time_sec
        record["profile_source"] = source
        record["bottleneck"] = label
        record["metrics"] = metrics
        record["feedback"] = feedback

        # R3 归因/蒸馏：源码特征 + 加速比，每轮成功都记录（无论瓶颈标签为何）
        feats = roofline.source_features(pure_code)
        record["source_features"] = feats
        effective_time = ncu_time_sec if ncu_time_sec is not None else time_sec
        record["speedup_vs_prev"] = eval_logic.compute_speedup(prev_time, effective_time)

        # 归因：由上一轮检索卡 + 前后两轮源码特征变化推断本轮实际应用了哪张卡
        applied_cards = []
        if prev_feats is not None and prev_retrieved:
            applied_cards = rag.infer_applied_cards(prev_retrieved, prev_feats, feats, cards)
        record["applied_cards"] = applied_cards

        # 置信度增量：应用卡按本轮是否更快给 +/-0.05（speedup>1 视为 improved）。
        # speedup 无法计算（上一轮无计时）时不做任何置信度更新，避免把「无法比较」
        # 误判成「变慢」而扣分。
        improved = record["speedup_vs_prev"] is not None and record["speedup_vs_prev"] > 1.0
        if applied_cards and record["speedup_vs_prev"] is not None:
            accumulate_confidence(applied_cards, improved, attempt + 1, applied=True,
                                  speedup=record["speedup_vs_prev"])

        # 画像后检索（RAG 设计 §3.3）：按瓶颈指纹 + 源码特征检索策略卡，
        # 供下一轮反馈注入；结果记入 record 的 retrieved_card_ids 证据链
        fb_hits = []
        knowledge_feedback = None
        if label in ("memory", "compute", "latency"):
            ncu_metrics = metrics if source == "ncu" else None
            fb_hits = retrieve_hits(bottleneck=label, source_features=feats, ncu_metrics=ncu_metrics)
            for h in fb_hits:
                retrieved_card_ids.add(h["card"]["id"])
            knowledge_feedback = rag.build_context_block(fb_hits) if fb_hits else None
        record["retrieved_card_ids"] = [h["card"]["id"] for h in fb_hits]

        # best-so-far 更新：仅当「正确且更快」才更新（更快优先用 ncu 可信计时）
        if first_ok_time is None:
            first_ok_time = effective_time
        if eval_logic.should_update_best(best["ok"], best["effective_time"], effective_time):
            best.update({
                "ok": True,
                "attempt": attempt + 1,
                "cu_filename": cu_filename,
                "binary": binary,
                "effective_time": effective_time,
                "self_reported_time": time_sec,
                "ncu_time_sec": ncu_time_sec,
                "max_diff": max_diff,
                "bottleneck": label,
                "profile_source": source,
                "metrics": metrics,
                "feedback": feedback,
            })
            best_source_features = feats
            best_applied_cards = applied_cards
            # P1（3.3）：best 输出矩阵单独落盘，避免被后续失败轮覆盖根目录 cuda_output_*.bin
            try:
                if os.path.exists(out_bin):
                    run_dir = os.path.join(RESULTS_DIR, str(run_id))
                    os.makedirs(run_dir, exist_ok=True)
                    shutil.copyfile(out_bin, os.path.join(run_dir, f"{matrix_size}_best.bin"))
            except Exception as e:
                print(f"[{matrix_size}] ⚠️ best 输出矩阵落盘失败（不影响评测）：{e}")
            if best["effective_time"] is not None:
                print(f"[{matrix_size}] ✅ 更新最优：attempt {attempt + 1}，耗时 {best['effective_time']:.6f}s")
            else:
                print(f"[{matrix_size}] ✅ 记录首个正确结果：attempt {attempt + 1}")

        records.append(record)

        # 更新下一轮归因所用的「上一成功轮」状态
        prev_feats = feats
        prev_retrieved = [h["card"]["id"] for h in fb_hits]
        prev_time = effective_time

        # 瓶颈识别 → 策略选择 → 自动迭代：memory/compute/latency 三类瓶颈都要接策略继续优化
        if label in ("memory", "compute", "latency") and attempt + 1 < max_retries:
            feedback_msg = feedback
            if knowledge_feedback:
                feedback_msg = feedback + "\n\n" + knowledge_feedback
            messages.append({"role": "assistant", "content": raw_text})
            messages.append({"role": "user", "content": feedback_msg})
            print(f"[{matrix_size}] 🔄 检测到 {label} 瓶颈，检索到 {len(fb_hits)} 张策略卡，已反馈给 AI 继续优化...")
            continue

        # 无瓶颈（unknown / 画像缺失）→ 没有有效策略可反馈，停止迭代
        break

    # 循环结束：只要有 best 就返回成功；后续 attempt 失败不会「反杀」已找到的最优
    timestamp = datetime.now(timezone.utc).isoformat()

    # R3 自蒸馏 + 置信度更新（仅知识库开启时；产物供人工 review 后合并）
    candidates_created = []
    knowledge_updates_applied = {}
    if kb["enabled"]:
        # 蒸馏判断：best 比第一次成功 attempt 更快（speedup>1）才可能产出新知识。
        speedup = eval_logic.compute_speedup(first_ok_time, best["effective_time"]) if best["ok"] else None
        distill_candidate = None
        if speedup is not None and speedup > 1.0:
            if best_applied_cards:
                # P1-1：提速来自应用**知识库里已有的卡** → 只追加 evidence（已随
                # confidence_events 记录），不再蒸馏重复候选卡，避免候选库堆满已知策略副本。
                print(
                    f"[{matrix_size}] 🧪 提速来自已应用策略卡 {best_applied_cards}，"
                    "跳过新候选卡蒸馏（证据已入 knowledge_updates）"
                )
            else:
                distill_candidate = rag.distill_from_run({
                    "ok": True,
                    "bottleneck": best.get("bottleneck"),
                    "source_features": best_source_features or {},
                    "speedup": speedup,
                    "run_id": run_id,
                    "matrix_size": matrix_size,
                    "attempt": best.get("attempt"),
                    "operator_spec": operator_spec,
                })

        # 置信度增量 → knowledge_updates_applied（纯逻辑，供 payload 与合并 CLI 使用）
        for cid, ev in confidence_events.items():
            cur = ev["current_confidence"]
            knowledge_updates_applied[cid] = {
                "delta": ev["delta"],
                "current_confidence": cur,
                "suggested_confidence": rag.clamp_confidence(cur + ev["delta"]),
                "rounds": ev["rounds"],
                "evidence": ev.get("evidence", []),
            }

        # 落盘（P1-3：整块 I/O 包 try/except，失败只告警，绝不阻断 persist_result）
        try:
            if distill_candidate:
                os.makedirs(CANDIDATES_DIR, exist_ok=True)
                # P1-2：文件名与候选卡 id 一致（id 已含 {N}-a{attempt}），避免跨尺寸互覆盖
                cand_path = os.path.join(CANDIDATES_DIR, f"{distill_candidate['id']}.json")
                with open(cand_path, "w", encoding="utf-8") as f:
                    json.dump(distill_candidate, f, ensure_ascii=False, indent=2, default=str)
                candidates_created.append(distill_candidate["id"])
                print(f"[{matrix_size}] 🧪 蒸馏候选卡 {distill_candidate['id']} → {cand_path}")

            if confidence_events:
                run_dir = os.path.join(RESULTS_DIR, str(run_id))
                os.makedirs(run_dir, exist_ok=True)
                # P0-1：文件名带尺寸，避免并发尺寸间互相覆盖同一 knowledge_updates.json
                updates_path = os.path.join(run_dir, f"knowledge_updates_{matrix_size}.json")
                with open(updates_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "run_id": run_id,
                        "matrix_size": matrix_size,
                        "updates": knowledge_updates_applied,
                    }, f, ensure_ascii=False, indent=2, default=str)
                print(f"[{matrix_size}] 📝 置信度更新已落盘：{updates_path}")
        except Exception as e:
            print(f"[{matrix_size}] ⚠️ R3 蒸馏/更新落盘失败（不影响评测结果）：{e}")

    knowledge = {
        "enabled": kb["enabled"],
        "mode": kb["mode"],
        "top_k": kb["top_k"],
        "cards_loaded": len(cards),
        "retrieved_card_ids": sorted(retrieved_card_ids),
        "candidates_created": candidates_created,
        "knowledge_updates_applied": knowledge_updates_applied,
    }
    if knowledge_warning:
        knowledge["warning"] = knowledge_warning
    payload = eval_logic.build_result_payload(
        run_id=run_id,
        timestamp=timestamp,
        matrix_size=matrix_size,
        model_name=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        best=best,
        records=records,
        env_snapshot=env_snapshot,
        knowledge=knowledge,
    )
    try:
        path, index_path = eval_logic.persist_result(payload, RESULTS_DIR)
        print(f"[{matrix_size}] 📁 证据链已落盘：{path}（索引 {index_path}）")
    except Exception as e:
        print(f"[{matrix_size}] ⚠️ 结果落盘失败（不影响评测）：{e}")

    if best["ok"]:
        if best["effective_time"] is not None:
            print(f"[{matrix_size}] ✅ 完成：最优 attempt {best['attempt']}，耗时 {best['effective_time']:.6f}s")
        else:
            print(f"[{matrix_size}] ✅ 完成：最优 attempt {best['attempt']}")
        return True, timings, ncu_timings
    print(f"[{matrix_size}] 🚨 达到最大重试次数，失败。")
    return False, timings, ncu_timings


async def main():
    # 尺寸由 SIZES 环境变量控制（逗号分隔，缺省 256,512,1024）。
    # P1：run_experiment 注入 SIZES，保证 smoke/分段实验只跑指定尺寸，不再写死三尺寸。
    try:
        sizes = [int(x) for x in os.getenv("SIZES", "256,512,1024").split(",") if x.strip()]
    except ValueError:
        print("[启动] ❌ SIZES 环境变量不是合法整数列表，退出。")
        return 1
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        print("[启动] ❌ SIZES 为空或全为非正数，退出。")
        return 1

    print("================ 开始并发评测 ================")
    if not await ensure_image():
        print("================ 沙箱镜像不可用，评测中止 ================")
        return 1

    _, ncu_msg = probe_host_ncu()
    print(f"[ncu] 宿主探测：{ncu_msg}")

    # run_id 可由 RUN_ID 环境变量覆盖（P0-3：批跑脚本按 run 注入确定性 id，便于归档）
    run_id = os.getenv("RUN_ID") or (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    )
    env_snapshot = collect_env_snapshot()
    print(f"[run] run_id={run_id}")

    # 使用 asyncio.gather 让这 3 个任务同时起飞；return_exceptions 保证单尺寸异常不毁掉整批
    tasks = [evaluate_matrix(size, run_id, env_snapshot) for size in sizes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for size, res in zip(sizes, results):
        if isinstance(res, Exception):
            print(f"[汇总] 尺寸 {size}: ❌异常 {res!r}")
        else:
            ok, _timings, _ncu_timings = res
            print(f"[汇总] 尺寸 {size}: {'✅成功' if ok else '❌失败'}")

    # 打印每个尺寸各次 attempt 的内核耗时列表（单位：秒），供画性能演进图。
    # 论文口径 3.1.3：以 ncu 可信计时（kernel-only）为准，自报 Time 作对照。
    def fmt(t):
        return "None" if t is None else f"{t:.6f}"

    print("\n========== 性能演进：ncu 可信计时（每次 attempt，秒） ==========")
    for size, res in zip(sizes, results):
        if isinstance(res, Exception):
            print(f"[{size}] [异常]")
            continue
        _ok, _timings, ncu_timings = res
        print(f"[{size}] [{', '.join(fmt(t) for t in ncu_timings)}]")
    print("\n========== 对照：自报 Time（每次 attempt，秒） ==========")
    for size, res in zip(sizes, results):
        if isinstance(res, Exception):
            print(f"[{size}] [异常]")
            continue
        _ok, timings, _ncu_timings = res
        print(f"[{size}] [{', '.join(fmt(t) for t in timings)}]")
    print("==============================================================")
    print("================ 评测全部结束 ================")

    # 任何尺寸异常/失败都记为整批失败，供批跑脚本按退出码重试（P0-3）
    all_ok = True
    for res in results:
        if isinstance(res, Exception) or not (res and res[0]):
            all_ok = False
            break
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
