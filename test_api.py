from openai import OpenAI
import re
import subprocess
import os
from dotenv import load_dotenv


load_dotenv() # 加载本地的 .env 文件

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com" 
)
messages = [
    {"role": "user", "content": "请用C++和CUDA写一个 1024*1024 的矩阵乘法代码。要求包含CPU计时，并打印出 'Time: xxx seconds'。不要任何废话，只要代码。"}
]

max_retries = 3  # 最多尝试 3 次（防止死循环）

for attempt in range(max_retries):
    print(f"\n==========================================")
    print(f"🚀 开始第 {attempt + 1} 次生成与评测...")
    print(f"==========================================")

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

    # 调用 Docker 沙箱（通过运行 auto_test.sh）
    result = subprocess.run(["bash", "auto_test.sh"], capture_output=True, text=True)
    
    # 检查评测结果
    if "Error" in result.stdout or "error" in result.stdout or "Traceback" in result.stderr or "Time:" not in result.stdout:
        print(f"❌ 第 {attempt + 1} 次评测失败！AI 生成的代码有 Bug。")
        print("正在收集错误日志，准备反馈给 AI 进行自我纠错...")
        
        # 把 AI 刚才写的烂代码和报错信息，追加到对话记录里
        messages.append({"role": "assistant", "content": raw_text})
        
        # 提取报错信息（截取终端输出的最后一部分，防止太长）
        error_feedback = f"你上次写的代码编译或运行失败了。终端输出如下：\n{result.stdout[-500:]}\n请仔细检查错误，修复后重新提供完整的 CUDA 代码。"
        messages.append({"role": "user", "content": error_feedback})
        
        print("🔄 正在将错误反馈发送给 AI，请稍候...")
        
    else:
        # 如果没有报错，且输出了时间
        print(f"🎉 第 {attempt + 1} 次评测成功！代码跑通！")
        print("\n--- 跑分结果 ---")
        # 只把带有 Time 的那行打印出来
        for line in result.stdout.split('\n'):
            if "Time:" in line or "耗时" in line:
                print(line)
        print("----------------\n")
        break  # 成功了就跳出循环

else:
    # 如果 3 次全失败了，会执行这里
    print("🚨 已达到最大重试次数，AI 依然未能生成正确代码。")


response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "请用C++和CUDA写一个1024*1024的矩阵乘法代码，要求包含生成随机数据的代码，使用CPU计时，并在最后打印出耗时。不要任何解释，只要代码"}
    ]
)