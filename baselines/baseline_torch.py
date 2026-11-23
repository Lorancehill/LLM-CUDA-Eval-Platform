import torch

N = 1024
WARMUP_ITERS = 5


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("没有可用的 CUDA GPU，torch baseline 需要 GPU。")

    device = torch.device("cuda")
    dtype = torch.float32

    # 统一确定性初始化（与 baseline.c 及 AI 生成核函数保持一致）：
    # A[i][j] = ((i+j)%10)*0.1, B[i][j] = ((i*3+j)%10)*0.1
    i = torch.arange(N, dtype=torch.int64, device=device)
    j = torch.arange(N, dtype=torch.int64, device=device)
    a = (((i[:, None] + j[None, :]) % 10).to(dtype) * 0.1)
    b = (((i[:, None] * 3 + j[None, :]) % 10).to(dtype) * 0.1)

    # 预热：跳过首次调用的 CUDA context / cuBLAS workspace 初始化开销
    for _ in range(WARMUP_ITERS):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()

    # GPU 精确计时：cudaEvent（与 CUDA 核函数计时同口径）
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"Torch matmul 计算 {N}x{N} 矩阵乘法完成。")
    print(f"校验值 C[0][0] = {c[0][0].item():.6f}")
    print(f"Torch matmul 耗时: {elapsed_ms:.2f} ms  (Time: {elapsed_ms / 1000.0:.6f} seconds)")
    print(f"设备: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
