"""矩阵校验工具：读取 C 基准 / CUDA 输出的 float32 二进制矩阵，用 numpy.allclose 校验。

二进制格式：raw float32、行优先（row-major）、N*N 个元素，无文件头。
与 baseline.c 的 fwrite 以及 prompt 要求的 fwrite 保持一致。
"""
import numpy as np

# 与 baseline.c / prompt 保持一致的确定性初始化
def init_matrices(N):
    i = np.arange(N, dtype=np.int64)
    j = np.arange(N, dtype=np.int64)
    A = ((i[:, None] + j[None, :]) % 10).astype(np.float32) * np.float32(0.1)
    B = ((i[:, None] * 3 + j[None, :]) % 10).astype(np.float32) * np.float32(0.1)
    return A, B


def reference_matmul(N):
    """numpy 计算的 float32 参考结果（async_eval 无 C 基准时作为基准）。"""
    A, B = init_matrices(N)
    return A @ B


def load_matrix(path, N):
    """读取 raw float32 行优先矩阵（N*N 个元素）。"""
    data = np.fromfile(path, dtype=np.float32)
    if data.size != N * N:
        raise ValueError(f"{path} 元素数 {data.size} != {N * N}，矩阵尺寸不符或写入不完整")
    return data.reshape(N, N)


def matrices_close(ref, got, rtol=1e-5, atol=1e-5):
    """返回 (是否通过, 最大绝对误差)。

    NaN/Inf、尺寸不符都判为失败（不通过）。
    rtol/atol 默认 1e-5：|ref - got| <= atol + rtol * |ref|。
    """
    if ref.shape != got.shape:
        return False, float("nan")
    if not np.all(np.isfinite(got)):
        return False, float("nan")
    return bool(np.allclose(ref, got, rtol=rtol, atol=atol)), float(np.max(np.abs(ref - got)))
