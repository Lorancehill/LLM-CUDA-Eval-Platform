from openai import OpenAI
import re
import subprocess
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 允许从 legacy/ 目录运行：把仓库根目录加入 sys.path，才能 import validate
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import validate


load_dotenv() # 加载本地的 .env 文件

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com" 
)
messages = [
    {"role": "user", "content": "请用C++和CUDA写一个 1024*1024 的矩阵乘法代码。 \
     要求用 wall-clock 计时测量 kernel 实际耗时（用 std::chrono::high_resolution_clock 或 CUDA event，不要用 clock()），并且必须定义一个CHECK宏来检查所有CUDA API的返回值（如cudaMalloc,cudaMemcpy,kernel launch等）， \
     并打印出 'Time: xxx seconds'。 \
     矩阵类型用 float（32位浮点），初始化 A[i][j] = ((i+j)%10)*0.1f，B[i][j] = ((i*3+j)%10)*0.1f，并打印 'C[0][0] = xxx' 用于结果校验。 \
     将结果矩阵 C（先 cudaMemcpy 从 device 拷回 host）以二进制写入文件 'cuda_output.bin'：fopen('cuda_output.bin','wb') 后 fwrite(C, sizeof(float), 1024*1024, fp)。 \
     计时必须准确！在开始计时前，请务必调用cudaDeviceSynchronize()确保之前的任务已完成；在核函数启动后、停止计时前，必须再次调用cudaDeviceSynchronize()。 \
     不要任何废话，只要代码。"}
]

max_retries = 3  # 最多尝试 3 次（防止死循环）

for attempt in range(max_retries):
    print("\n==========================================")
    print(f"🚀 开始第 {attempt + 1} 次生成与评测...")
    print("==========================================")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages
    )
    raw_text = response.choices[0].message.content

    # 正则清洗
    pattern = r'```(?:cpp|c\+\+|c)?\s*(.*?)\s*```'
    match = re.search(pattern, raw_text, re.DOTALL)
    
    if match:
        pure_code = match.group(1)
    else:
        pure_code = raw_text
        
    # 写入文件
    with open("matrix_mul.cu", "w") as f:
        f.write(pure_code)
    print("✅ 代码已写入 matrix_mul.cu，准备送入沙箱测试...")

    # 清理上次的 CUDA 输出，避免读到过期矩阵
    if os.path.exists("cuda_output.bin"):
        os.remove("cuda_output.bin")

    # 调用 Docker 沙箱（通过运行 auto_test.sh：C 基准 + CUDA）
    result = subprocess.run(["bash", "auto_test.sh"], capture_output=True, text=True)

    # 读取 C 基准与 CUDA 输出矩阵，用 numpy.allclose 做 1e-5 容差校验
    N = 1024
    if not os.path.exists("c_output.bin"):
        raise SystemExit("c_output.bin 缺失：C 基准编译/运行失败，请检查 baseline.c / auto_test.sh。")

    ok = False
    max_diff = None
    try:
        ref = validate.load_matrix("c_output.bin", N)
        if os.path.exists("cuda_output.bin"):
            got = validate.load_matrix("cuda_output.bin", N)
            ok, max_diff = validate.matrices_close(ref, got, rtol=1e-5, atol=1e-5)
    except ValueError as e:
        ok = False
        print(f"⚠️ 矩阵读取/校验异常：{e}")

    if not ok:
        print(f"❌ 第 {attempt + 1} 次评测失败！")
        messages.append({"role": "assistant", "content": raw_text})
        if max_diff is None:
            # 编译/运行失败：没有产生有效的 cuda_output.bin
            error_feedback = (
                f"你上次写的代码编译或运行失败，没有输出结果矩阵 cuda_output.bin。\n"
                f"[stdout 尾部]\n{result.stdout[-300:]}\n"
                f"[stderr 尾部]\n{result.stderr[-300:]}\n"
                f"请仔细检查错误，修复后重新提供完整的 CUDA 代码。"
            )
        else:
            # 有输出但结果与基准不符
            error_feedback = (
                f"你上次写的代码能运行，但结果矩阵与 C 基准不符（最大绝对误差 {max_diff:.6g}，"
                f"超过 1e-5 容差）。\n请检查矩阵乘法逻辑（索引、累加、边界），修复后重新提供完整的 CUDA 代码。"
            )
        messages.append({"role": "user", "content": error_feedback})
        print("🔄 正在将错误反馈发送给 AI，请稍候...")
        
    else:
        print(f"🎉 第 {attempt + 1} 次评测成功！结果与 C 基准一致（max diff {max_diff:.3g}）。")
        print("\n--- 跑分结果 ---")
        for line in result.stdout.split('\n'):
            if "Time:" in line or "耗时" in line:
                print(line)
        print("----------------\n")
        break  # 成功了就跳出循环

else:
    # 如果 3 次全失败了，会执行这里
    print("🚨 已达到最大重试次数，AI 依然未能生成正确代码。")