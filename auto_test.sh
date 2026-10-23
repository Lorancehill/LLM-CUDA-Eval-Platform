#!/bin/bash
# 这是一个大创项目初期的自动化评测脚本

echo "========== 开始评测 =========="

echo "[1/2] 正在编译并运行纯 C 基准代码..."
gcc baseline.c -o baseline -O3
echo "纯 C 耗时："
./baseline

echo "--------------------------------"

echo "[2/2] 正在将 AI 生成的代码送入 Docker 沙箱评测..."
docker run --rm -v $(pwd):/app --gpus all \
  --memory="2g" \
  --cpus="1.0" \
  --network none \
  nvidia/cuda:12.0.0-devel-ubuntu22.04 \
  bash -c "cd /app && timeout 15s nvcc matrix_mul.cu -o matrix_mul && echo  'CUDA 耗时：' && timeout 15s ./matrix_mul"

echo "========== 评测结束 =========="
