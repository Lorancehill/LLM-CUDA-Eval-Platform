#!/bin/bash
# 早期（大创初期）自动化评测脚本，已弃用；保留作历史参考。
# 当前主流程请用：python preflight.py && python run_experiment.py --exp exp1 ...

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========== 开始评测（legacy） =========="

echo "[1/3] 正在编译并运行纯 C 基准代码..."
gcc baselines/baseline_C.c -o /tmp/legacy_baseline -O3
echo "纯 C 耗时："
/tmp/legacy_baseline

echo "--------------------------------"

echo "[2/3] 正在将 AI 生成的代码送入 Docker 沙箱评测..."
docker run --rm -v "$(pwd)":/app --gpus all \
  --memory="2g" \
  --cpus="1.0" \
  --network none \
  nvidia/cuda:12.8.2-devel-ubuntu22.04 \
  bash -c "cd /app && timeout 15s nvcc -arch=sm_120 matrix_mul.cu -o matrix_mul && echo  'CUDA 耗时：' && timeout 15s ./matrix_mul"

echo "--------------------------------"

echo "[3/3] 正在运行 torch.matmul 基准代码..."
python baselines/baseline_torch.py

echo "========== 评测结束 =========="
