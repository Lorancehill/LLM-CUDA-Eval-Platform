import asyncio
import re
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY"), 
    base_url = "https://api.deepseek.com" 
)

async def evaluate_matrix(matrix_size):
    messages = [
        {"role": "user", "content": f"请用C++和CUDA写一个 {matrix_size}*{matrix_size} 的矩阵乘法代码。要求包含CPU计时，并打印出 'Time: xxx seconds'。不要任何废话，只要代码。"}
    ]
    max_retries = 3

    for attempt in range(max_retries):
        print(f"\n[{matrix_size}] ==========================================")
        print(f"[{matrix_size}] 🚀 开始第 {attempt + 1} 次生成与评测...")

        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages
        )
        raw_text = response.choices[0].message.content

        pattern = r'```(?:cpp|c\+\+|c)?\s*(.*?)\s*```'
        match = re.search(pattern, raw_text, re.DOTALL)
        pure_code = match.group(1) if match else raw_text
            
        cu_filename = f"matrix_mul_{matrix_size}.cu"
        with open(cu_filename, "w") as f:
            f.write(pure_code)
        print(f"[{matrix_size}] ✅ 代码已写入 {cu_filename}，准备送入沙箱测试...")

        # 这里加了 --rm 自动清理容器，并执行编译和运行
        cmd = f"docker run --rm -memory='2g' --cpus='1.0' --network none -v $(pwd):/app --gpus all nvidia/cuda:12.0.0-devel-ubuntu22.04 bash -c 'cd /app && nvcc {cu_filename} -o test_{matrix_size} && ./test_{matrix_size}'"
        
        # 异步执行外部命令
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # 提前给变量赋初值，防止 NameError
        stdout, stderr = b"", b""

        try:
            # 强制设置 15 秒超时
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        except asyncio.TimeoutError:
            # 如果超时，强制杀掉容器进程
            try:
                process.kill()
            except:
                pass # 防止进程已经死了导致 kill 报错
            print(f"[{matrix_size}] ❌ 运行超时，已被系统强杀！")
            continue # 直接进入下一次重试循环
        except Exception as e:
            # 如果发生其他未知错误，也捕捉住，防止整个程序崩溃
            print(f"[{matrix_size}] ❌ 发生未知错误: {e}")
            continue

        # 只有 try 成功，或者捕获了非 Timeout 的普通异常后，才会走到这里
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        # 检查评测结果
        if "Error" in stdout_str or "error" in stdout_str or "Traceback" in stderr_str or "Time:" not in stdout_str:
            print(f"[{matrix_size}] ❌ 第 {attempt + 1} 次评测失败！正在反馈给 AI...")
            
            messages.append({"role": "assistant", "content": raw_text})
            error_feedback = f"你上次写的代码编译或运行失败了。终端输出如下：\n{stdout_str[-500:]}\n请仔细检查错误，修复后重新提供完整的 CUDA 代码。"
            messages.append({"role": "user", "content": error_feedback})
            
        else:
            print(f"[{matrix_size}] 🎉 第 {attempt + 1} 次评测成功！")
            for line in stdout_str.split('\n'):
                if "Time:" in line or "耗时" in line:
                    print(f"[{matrix_size}] --- 跑分: {line.strip()}")
            return True # 成功了就返回 True，结束当前任务

    print(f"[{matrix_size}] 🚨 达到最大重试次数，失败。")
    return False

async def main():
    # 我们同时测试 3 种不同的矩阵大小
    sizes = [256, 512, 1024]

    #限制同时最多只有一个任务传占用GPU   
    semaphore = asyncio.Semaphore()

    #包装任务
    async def run_with_semaphore(size):
        async with semaphore:
            return await evaluate_matrix(size)
    
    print("================ 开始并发评测 ================")
    # 使用 asyncio.gather 让这 3 个任务同时起飞
    tasks = [run_with_semaphore(size) for size in sizes]
    await asyncio.gather(*tasks)
    print("================ 评测全部结束 ================")

if __name__ == "__main__":
    asyncio.run(main())