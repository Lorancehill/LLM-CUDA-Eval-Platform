"""矩阵校验工具：读取 C 基准 / CUDA 输出的 float32 二进制矩阵，用 numpy.allclose 校验。

二进制格式：raw float32、行优先（row-major）、N*N 个元素，无文件头。
与 baseline.c 的 fwrite 以及 prompt 要求的 fwrite 保持一致。
"""
import numpy as np

# float32 机器精度（用于按尺寸校准比较容差）
EPS32 = float(np.finfo(np.float32).eps)  # ≈1.19e-7


# 与 baseline.c / prompt 保持一致的确定性初始化（float64：0~9 整数精确可表示）
def init_matrices_float64(N):
    i = np.arange(N, dtype=np.int64)
    j = np.arange(N, dtype=np.int64)
    A = ((i[:, None] + j[None, :]) % 10).astype(np.float64) * 0.1
    B = ((i[:, None] * 3 + j[None, :]) % 10).astype(np.float64) * 0.1
    return A, B


def reference_matmul(N):
    """float64 参考结果（async_eval 无 C 基准时作为基准）。

    用 float64 计算参考，避免 numpy float32 参考自身的舍入误差叠加到判据上
    （B5：2048 时 float32 参考 + 固定 1e-5 会把正确内核误判为错误）。
    """
    A, B = init_matrices_float64(N)
    return A @ B


def default_tolerance(N):
    """按尺寸校准的比较容差 (rtol, atol)。

    float32 顺序累加舍入误差近似随 N 线性增长（gamma_N ≈ N*eps32），固定 1e-5
    在 2048 会把正确内核误判为错误。这里 rtol 随 N 线性放大，atol 只兜底零附近。
    """
    rtol = max(1e-6, 2.0 * N * EPS32)
    return rtol, 1e-6


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
