import asyncio
import json
import os
import re
import subprocess
from openai import AsyncOpenAI
from dotenv import load_dotenv
import validate
import profiler

load_dotenv()

client = AsyncOpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"), 
    base_url = "https://api.deepseek.com" 
)

#限制同时最多只有一个任务传占用GPU   
gpu_sem = asyncio.Semaphore(1)

def parse_time_seconds(stdout_str):
    """从 CUDA 输出解析 'Time: xxx seconds' 中的秒数；无匹配返回 None。"""
    m = re.search(r"Time:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", stdout_str)
    return float(m.group(1)) if m else None


def should_update_best(best_ok, best_time, new_time):
    """best-so-far 更新判据（纯函数，便于单测）。

    - 尚无正确结果（best_ok=False）：无条件记录为 best（首个正确结果）。
    - 已有正确结果：仅当新旧耗时都可测量且新耗时严格更小才替换；
      否则（任一耗时缺失，无法证明更快）保留旧 best，避免被改坏的内核「反杀」。
    """
    if not best_ok:
        return True
    return best_time is not None and new_time is not None and new_time < best_time

# Docker 沙箱运行超时上限（秒）
TIMEOUT_SECONDS = 20
# ncu 画像超时上限（秒）：ncu 需多次 replay kernel 采集全部指标，比普通运行慢得多
NCU_TIMEOUT_SECONDS = 120

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
        if r.returncode == 0:
            print("[镜像] ✅ 拉取完成。")
            return True
        print(f"[镜像] ⚠️ 拉取失败：{r.stderr.decode(errors='replace')[-300:]}")
        if attempt < PULL_RETRIES:
            print("[镜像] 断点续传重试中...")
            continue
    print(f"[镜像] ❌ 拉取 {PULL_RETRIES} 次均失败。请检查网络后手动执行：docker pull {CUDA_IMAGE}")
    return False


async def _profile_kernel(matrix_size, container_name, binary, kernel_name=None):
    """在沙箱内用 ncu 抓取硬件画像；失败/无指标返回 None（不影响主流程）。

    binary 为本次 attempt 独立编译出的可执行文件名；kernel_name 用于多 kernel
    代码时只 profile 目标内核（避免 init/融合算子等被误采为第一个 kernel）。
    """
    cmd = (
        f"docker run --rm --name {container_name} --memory=2g --cpus=1.0 --network none "
        f"-v $(pwd):/app --gpus all {CUDA_IMAGE} "
        f"bash -c 'cd /app && {profiler.build_ncu_cmd(binary, kernel_name)}'"
    )

    # ncu 画像也占用 GPU，串行执行
    async with gpu_sem:
        try:
            _, stdout, stderr, _ = await _docker_run(
                cmd, container_name, timeout=NCU_TIMEOUT_SECONDS
            )
        except Exception as e:
            print(f"[{matrix_size}] ⚠️ ncu 画像失败（Docker 异常）: {e}")
            return None

    stdout_str = stdout.decode(errors="replace")
    stderr_str = stderr.decode(errors="replace")
    metrics = profiler.parse_ncu_csv(stdout_str)
    if not metrics:
        # 抓不到指标：ncu 失败 / 芯片不支持 / 未安装，降级为无画像，不影响评测
        tail = (stderr_str or stdout_str).strip()[-200:]
        print(f"[{matrix_size}] ⚠️ ncu 未抓到硬件指标，跳过画像。{tail}")
        return None
    return metrics


def _persist_result(matrix_size, best, records):
    """把每轮画像指标/分类/反馈与 best-so-far 落盘，便于事后回溯「瓶颈→策略→收益」证据链。

    落盘失败（磁盘满/权限）只告警，不阻断评测主流程。
    """
    def _clean_max_diff(v):
        # NaN -> None，避免写出非法 JSON（json.dump 默认会写成 NaN）
        return None if (v is not None and v != v) else v

    payload = {
        "matrix_size": matrix_size,
        "success": bool(best["ok"]),
        "best": {
            "attempt": best["attempt"],
            "cu_filename": best["cu_filename"],
            "binary": best["binary"],
            "effective_time": best["effective_time"],
            "self_reported_time": best["self_reported_time"],
            "ncu_time_sec": best["ncu_time_sec"],
            "max_diff": _clean_max_diff(best["max_diff"]),
            "bottleneck": best["bottleneck"],
            "metrics": best["metrics"],
            "feedback": best["feedback"],
        },
        "attempts": [
            {**r, "max_diff": _clean_max_diff(r["max_diff"])}
            for r in records
        ],
    }
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, f"{matrix_size}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"[{matrix_size}] 📁 画像/分类/反馈证据链已落盘：{path}")
    except Exception as e:
        print(f"[{matrix_size}] ⚠️ 结果落盘失败（不影响评测）：{e}")


async def evaluate_matrix(matrix_size):
    messages = [
        {"role": "user", "content": f"请用C++和CUDA写一个 {matrix_size}*{matrix_size} 的矩阵乘法代码。要求：\n1. 矩阵类型用 float（32位浮点），初始化 A[i][j] = ((i+j)%10)*0.1f，B[i][j] = ((i*3+j)%10)*0.1f；\n2. 用 wall-clock 测量 kernel 执行耗时（用 std::chrono::high_resolution_clock 或 CUDA event，在 kernel 启动前后各调一次 cudaDeviceSynchronize；不要用 clock()，它测的是 CPU 时间），并打印出 'Time: xxx seconds'；\n3. 打印 'C[0][0] = xxx' 用于结果校验；\n4. 将结果矩阵 C（先 cudaMemcpy 从 device 拷回 host）以二进制写入文件 'cuda_output_{matrix_size}.bin'：fopen('cuda_output_{matrix_size}.bin','wb') 后 fwrite(C, sizeof(float), {matrix_size}*{matrix_size}, fp)。\n不要任何废话，只要代码。"}
    ]
    max_retries = 3

    # 参考结果（确定性初始化，numpy float32 计算一次）
    ref = validate.reference_matmul(matrix_size)
    out_bin = f"cuda_output_{matrix_size}.bin"

    # 记录每一次 attempt 的内核耗时（秒）；无输出（超时/编译失败）记 None
    timings = []

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
        "metrics": None,
        "feedback": None,
    }
    # 每轮结构化记录，落盘用于回溯「瓶颈 → 策略 → 收益」证据链
    records = []

    for attempt in range(max_retries):
        print(f"\n[{matrix_size}] ==========================================")
        print(f"[{matrix_size}] 🚀 开始第 {attempt + 1} 次生成与评测...")

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        raw_text = response.choices[0].message.content

        #正则清洗
        pattern = r'```(?:cpp|c\+\+|c)?\s*(.*?)\s*```'
        match = re.search(pattern, raw_text, re.DOTALL)
        pure_code = match.group(1) if match else raw_text
            
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
        cmd = f"docker run --rm --name {container_name} --memory=2g --cpus=1.0 --network none -v $(pwd):/app --gpus all {CUDA_IMAGE} bash -c 'cd /app && nvcc -arch=sm_120 {cu_filename} -o {binary} && ./{binary}'"
        
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

        # 记录本次 attempt 的内核耗时（秒）；无 'Time:' 输出记 None
        time_sec = parse_time_seconds(stdout_str)
        timings.append(time_sec)

        # 读取 CUDA 输出矩阵，与 numpy 参考做 1e-5 容差校验
        ok = False
        max_diff = None
        if os.path.exists(out_bin):
            try:
                got = validate.load_matrix(out_bin, matrix_size)
                ok, max_diff = validate.matrices_close(ref, got, rtol=1e-5, atol=1e-5)
            except ValueError as e:
                print(f"[{matrix_size}] ⚠️ 矩阵读取/校验异常: {e}")

        record = {
            "attempt": attempt + 1,
            "ok": ok,
            "time_sec": time_sec,
            "ncu_time_sec": None,
            "max_diff": max_diff,
            "cu_filename": cu_filename,
            "binary": binary,
            "out_bin": out_bin,
            "bottleneck": None,
            "metrics": None,
            "feedback": None,
        }

        if not ok:
            print(f"[{matrix_size}] ❌ 第 {attempt + 1} 次评测失败！正在反馈给 AI...")
            messages.append({"role": "assistant", "content": raw_text})

            # 真正的自我纠错：把报错信息截取最后 300 个字符存进变量，喂回 DeepSeek
            err_tail = stderr_str[-300:].strip()
            out_tail = stdout_str[-300:].strip()

            if timed_out:
                error_feedback = (
                    f"你上次的 CUDA 代码在 {TIMEOUT_SECONDS} 秒内未跑完（超时被强杀）。\n"
                    f"[运行输出尾部]\n{out_tail}\n"
                    f"[错误输出尾部]\n{err_tail}\n"
                    f"可能是算法太慢（如 naive 三重循环）或存在死循环，请优化 kernel 后重新提供完整代码。"
                )
            elif returncode is None or returncode != 0:
                error_feedback = (
                    f"你的代码在 Docker 沙箱中运行失败（退出码 {returncode}）。\n"
                    f"[stdout 尾部]\n{out_tail}\n"
                    f"[stderr 尾部]\n{err_tail}\n"
                    f"请根据报错信息定位问题，修复后重新提供完整的 CUDA 代码。"
                )
            elif max_diff is None:
                error_feedback = (
                    f"你上次写的代码能编译运行，但没有输出结果矩阵 {out_bin}。\n"
                    f"[stdout 尾部]\n{out_tail}\n"
                    f"[stderr 尾部]\n{err_tail}\n"
                    f"请确认 fwrite 写入了 {out_bin}，修复后重新提供完整的 CUDA 代码。"
                )
            else:
                error_feedback = (
                    f"你上次写的代码能运行，但结果矩阵与基准不符（最大绝对误差 {max_diff:.6g}，"
                    f"超过 1e-5 容差）。\n请检查矩阵乘法逻辑（索引、累加、边界），修复后重新提供完整的 CUDA 代码。"
                )
            messages.append({"role": "user", "content": error_feedback})
            records.append(record)
            continue
        else:
            print(f"[{matrix_size}] 🎉 第 {attempt + 1} 次评测成功！结果与基准一致（max diff {max_diff:.3g}）。")
            for line in stdout_str.split('\n'):
                if "Time:" in line or "耗时" in line:
                    print(f"[{matrix_size}] --- 跑分: {line.strip()}")

            # 结果正确后，用 Nsight Compute 抓取硬件画像，从「只看耗时」升级为「瓶颈分类」
            profile = await _profile_kernel(matrix_size, f"{container_name}_ncu", binary)
            label = None
            feedback = None
            ncu_time_sec = None
            if profile is not None:
                print(f"[{matrix_size}] 📊 硬件画像：\n{profiler.format_metrics(profile)}")
                label, feedback = profiler.classify_bottleneck(profile)
                print(f"[{matrix_size}] 🧭 瓶颈分类: {label}")
                # ncu 的 gpu__time_duration.sum 是可信内核计时，优先用它替代模型自报的 Time:
                if profile.get("gpu__time_duration.sum") is not None:
                    ncu_time_sec = profile["gpu__time_duration.sum"] / 1e9
                    print(f"[{matrix_size}] ⏱️  ncu 可信计时 {ncu_time_sec:.6f}s（自报 {time_sec if time_sec is not None else 'N/A'}s）")
            record["ncu_time_sec"] = ncu_time_sec
            record["bottleneck"] = label
            record["metrics"] = profile
            record["feedback"] = feedback

            # best-so-far 更新：仅当「正确且更快」才更新（更快优先用 ncu 可信计时）
            effective_time = ncu_time_sec if ncu_time_sec is not None else time_sec
            if should_update_best(best["ok"], best["effective_time"], effective_time):
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
                    "metrics": profile,
                    "feedback": feedback,
                })
                if best["effective_time"] is not None:
                    print(f"[{matrix_size}] ✅ 更新最优：attempt {attempt + 1}，耗时 {best['effective_time']:.6f}s")
                else:
                    print(f"[{matrix_size}] ✅ 记录首个正确结果：attempt {attempt + 1}")

            records.append(record)

            # 瓶颈识别 → 策略选择 → 自动迭代：memory/compute/latency 三类瓶颈都要接策略继续优化
            if label in ("memory", "compute", "latency") and attempt + 1 < max_retries:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": feedback})
                print(f"[{matrix_size}] 🔄 检测到 {label} 瓶颈，已反馈给 AI 继续优化...")
                continue

            # 无瓶颈（unknown / 画像缺失）→ 没有有效策略可反馈，停止迭代
            break

    # 循环结束：只要有 best 就返回成功；后续 attempt 失败不会「反杀」已找到的最优
    _persist_result(matrix_size, best, records)
    if best["ok"]:
        if best["effective_time"] is not None:
            print(f"[{matrix_size}] ✅ 完成：最优 attempt {best['attempt']}，耗时 {best['effective_time']:.6f}s")
        else:
            print(f"[{matrix_size}] ✅ 完成：最优 attempt {best['attempt']}")
        return True, timings
    print(f"[{matrix_size}] 🚨 达到最大重试次数，失败。")
    return False, timings

async def main():
    # 我们同时测试 3 种不同的矩阵大小
    sizes = [256, 512, 1024]
    
    print("================ 开始并发评测 ================")
    if not await ensure_image():
        print("================ 沙箱镜像不可用，评测中止 ================")
        return
    # 使用 asyncio.gather 让这 3 个任务同时起飞
    tasks = [evaluate_matrix(size) for size in sizes]
    results = await asyncio.gather(*tasks)
    for size, (ok, _timings) in zip(sizes, results):
        print(f"[汇总] 尺寸 {size}: {'✅成功' if ok else '❌失败'}")

    # 打印每个尺寸各次 attempt 的内核耗时列表（单位：秒），供画性能演进图
    def fmt(t):
        return "None" if t is None else f"{t:.6f}"
    print("\n================ 性能演进（每次 attempt 耗时，单位：秒） ================")
    for size, (_ok, timings) in zip(sizes, results):
        print(f"[{size}] [{', '.join(fmt(t) for t in timings)}]")
    print("======================================================================")
    print("================ 评测全部结束 ================")

if __name__ == "__main__":
    asyncio.run(main())