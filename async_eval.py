import asyncio
import re
import os
import subprocess
from openai import AsyncOpenAI
from dotenv import load_dotenv
import validate

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

# Docker 沙箱运行超时上限（秒）
TIMEOUT_SECONDS = 20


async def _docker_run(cmd, container_name):
    """在 TIMEOUT_SECONDS 内运行 docker 命令；超时则强杀容器并返回已捕获的部分输出。

    返回 (returncode, stdout_bytes, stderr_bytes, timed_out)：
    - 正常结束：returncode 为进程退出码，timed_out=False；
    - 超时：returncode=None，stdout/stderr 为超时前已捕获的部分输出，timed_out=True。
    """
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd, shell=True, capture_output=True, timeout=TIMEOUT_SECONDS
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

async def evaluate_matrix(matrix_size):
    messages = [
        {"role": "user", "content": f"请用C++和CUDA写一个 {matrix_size}*{matrix_size} 的矩阵乘法代码。要求：\n1. 矩阵类型用 float（32位浮点），初始化 A[i][j] = ((i+j)%10)*0.1f，B[i][j] = ((i*3+j)%10)*0.1f；\n2. 包含CPU计时，并打印出 'Time: xxx seconds'；\n3. 打印 'C[0][0] = xxx' 用于结果校验；\n4. 将结果矩阵 C（先 cudaMemcpy 从 device 拷回 host）以二进制写入文件 'cuda_output_{matrix_size}.bin'：fopen('cuda_output_{matrix_size}.bin','wb') 后 fwrite(C, sizeof(float), {matrix_size}*{matrix_size}, fp)。\n不要任何废话，只要代码。"}
    ]
    max_retries = 3

    # 参考结果（确定性初始化，numpy float32 计算一次）
    ref = validate.reference_matmul(matrix_size)
    out_bin = f"cuda_output_{matrix_size}.bin"

    # 记录每一次 attempt 的内核耗时（秒）；无输出（超时/编译失败）记 None
    timings = []

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
            
        cu_filename = f"matrix_mul_{matrix_size}.cu"
        with open(cu_filename, "w") as f:
            f.write(pure_code)
        print(f"[{matrix_size}] ✅ 代码已写入 {cu_filename}，准备送入沙箱测试...")

        # 清理上次的输出，避免读到过期矩阵
        if os.path.exists(out_bin):
            os.remove(out_bin)

        # 这里加了 --rm 自动清理容器，并执行编译和运行
        container_name = f"test_{matrix_size}_{attempt}"
        cmd = f"docker run --rm --name {container_name} --memory=2g --cpus=1.0 --network none -v $(pwd):/app --gpus all nvidia/cuda:12.0.0-devel-ubuntu22.04 bash -c 'cd /app && nvcc {cu_filename} -o test_{matrix_size} && ./test_{matrix_size}'"
        
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
        timings.append(parse_time_seconds(stdout_str))

        # 读取 CUDA 输出矩阵，与 numpy 参考做 1e-5 容差校验
        ok = False
        max_diff = None
        if os.path.exists(out_bin):
            try:
                got = validate.load_matrix(out_bin, matrix_size)
                ok, max_diff = validate.matrices_close(ref, got, rtol=1e-5, atol=1e-5)
            except ValueError as e:
                print(f"[{matrix_size}] ⚠️ 矩阵读取/校验异常: {e}")

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
        else:
            print(f"[{matrix_size}] 🎉 第 {attempt + 1} 次评测成功！结果与基准一致（max diff {max_diff:.3g}）。")
            for line in stdout_str.split('\n'):
                if "Time:" in line or "耗时" in line:
                    print(f"[{matrix_size}] --- 跑分: {line.strip()}")
            return True, timings # 成功了就返回 True 和耗时列表，结束当前任务

    print(f"[{matrix_size}] 🚨 达到最大重试次数，失败。")
    return False, timings

async def main():
    # 我们同时测试 3 种不同的矩阵大小
    sizes = [256, 512, 1024]
    
    print("================ 开始并发评测 ================")
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