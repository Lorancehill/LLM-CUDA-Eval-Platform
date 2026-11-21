"""
矩阵校验工具：读取 C 基准 / CUDA 输出的 float32 二进制矩阵，用 numpy.allclose 校验。

二进制格式：raw float32、行优先（row-major）、N*N 个元素，无文件头。
与 baseline.c 的 fwrite 以及 prompt 要求的 fwrite 保持一致。
"""
import os
import numpy as np

__all__ = [
    "EPS32",
    "init_matrices_float64",
    "reference_matmul",
    "default_tolerance",
    "load_matrix",
    "matrices_close",
    "matrices_close_details",
]

EPS32 = float(np.finfo(np.float32).eps)  # float32 机器精度（用于按尺寸校准比较容差）, ≈1.1920929e-7

def _validate_size(N: int) -> None :
    """
    校验矩阵尺寸参数：必须是 >= 1 的整数（N <= 0 属未定义输入）。

    内部辅助函数，外部不直接调用。
    """
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N < 1 :
        raise ValueError(f"N 必须是 >= 1 的整数，收到 {N!r}")

def init_matrices_float64(N: int) -> tuple[np.ndarray, np.ndarray] :
    """
    生成确定性输入矩阵 A、B（float64，形状 (N, N)）。

    与 baseline_C.c / prompt 保持一致的确定性初始化（float64：0~9 整数精确可表示）。
    """
    _validate_size(N)

    i = np.arange(N, dtype = np.int64)
    j = np.arange(N, dtype = np.int64)

    A = ((i[:, None] + j[None, : ]) % 10).astype(np.float64) * 0.1
    B = ((i[ : , None] * 3 + j[None, : ]) % 10).astype(np.float64) * 0.1

    return A, B


def reference_matmul(N: int) -> np.ndarray :
    """
    float64 参考结果（async_eval 无 C 基准时作为基准）。

    用 float64 计算参考，避免 numpy float32 参考自身的舍入误差叠加到判据上
    （B5：2048 时 float32 参考 + 固定 1e-5 会把正确内核误判为错误）。
    """
    A, B = init_matrices_float64(N)

    return A @ B


def default_tolerance(N: int) -> tuple[float, float] :
    """
    按尺寸校准的比较容差 (rtol, atol)。

    float32 顺序累加舍入误差近似随 N 线性增长（gamma_N ≈ N * eps32），固定 1e-5
    在 2048 会把正确内核误判为错误。这里 rtol 随 N 线性放大，atol 只兜底零附近。
    """
    _validate_size(N)

    rtol = max(1e-6, 2.0 * EPS32 * N)

    return rtol, 1e-6


def load_matrix(path, N: int) -> np.ndarray :
    """
    读取 raw float32 行优先矩阵（恰好 N*N 个元素）。

    Args:
        path: 文件路径（str / bytes / os.PathLike）。
        N: 矩阵阶数，必须 >= 1。

    Raises:
        ValueError: 文件不可读、字节数 != N*N*4、或元素数 != N*N。
    """
    _validate_size(N)
    
    expected_elems = N * N
    expected_bytes = expected_elems * 4

    try :
        actual_bytes = os.path.getsize(path)
    except (OSError, TypeError) as exc :
        raise ValueError(f"无法读取矩阵文件 {path!r}: {exc}") from exc
    if actual_bytes != expected_bytes :
        raise ValueError(
            f"{path} 文件大小 {actual_bytes} 字节 != 期望 {expected_bytes} 字节"
            f"（N={N} 的 float32 矩阵），矩阵尺寸不符或写入不完整"
        )

    try :
        data = np.fromfile(path, dtype = np.float32)
    except OSError as exc :
        raise ValueError(f"无法读取矩阵文件 {path!r}: {exc}") from exc
    if data.size != expected_elems :
        raise ValueError(
            f"{path} 元素数 {data.size} != {expected_elems}，矩阵尺寸不符或写入不完整"
        )
    return data.reshape(N, N)


def matrices_close(ref: np.ndarray, got: np.ndarray,
                   rtol: float | None = None, atol: float | None = None) -> tuple[bool, float] :
    """
    返回 (是否通过, 最大绝对误差)。

    NaN/Inf、尺寸不符都判为失败（不通过）；空矩阵显式抛 ValueError（N >= 1 假设）。

    rtol/atol 传 None 时回退到旧默认 1e-5（只保证 API 兼容）；生产路径请显式传
    "default_tolerance(N)"——本函数拿不到 N，无法可靠推导校准容差。

    判据："np.allclose(ref, got, rtol=rtol, atol=atol)"，即
    "|ref - got| <= atol + rtol * |got|"（NumPy 的 rtol 乘第二个参数；本项目
    ref 与 got 近似相等，与文档里写作 "|ref|" 只差二阶小量）。
    """
    if rtol is None :
        rtol = 1e-5
    if atol is None :
        atol = 1e-5
    if ref.shape != got.shape :
        return False, float("nan")
    if ref.size == 0 :
        raise ValueError("空矩阵（任一维为 0）不支持，N 应 >= 1")
    if not np.all(np.isfinite(got)) :
        return False, float("nan")
    return (
        bool(np.allclose(ref, got, rtol = rtol, atol = atol)),
        float(np.max(np.abs(ref - got))),
    )

def matrices_close_details(ref: np.ndarray, got: np.ndarray,
                           rtol: float | None = None, atol: float | None = None) -> dict :
    """
    辅助接口：在 "matrices_close" 基础上给出可解释的误差画像。

    返回 dict（全部为 Python 原生类型，可直接 JSON 落盘）：
        ok:            bool，判定是否通过；
        max_diff:      float，最大绝对误差（= matrices_close 的第二返回值）；
        max_ratio:     float，"max(|ref - got| / (atol + rtol*|got|))"，>= 1 即失败；
        n_violations:  int，超阈值元素个数；
        worst_index:   tuple[int, ...] 或 None，max_ratio 所在下标。

    rtol/atol 为 None 时与 "matrices_close" 一致回退到 1e-5；生产路径显式传
    "default_tolerance(N)"。该函数不改变 "matrices_close" 的既有契约，
    供 dashboard / 错误反馈可选使用。
    """
    ok, max_diff = matrices_close(ref, got, rtol = rtol, atol = atol)
    if ref.shape != got.shape or not np.all(np.isfinite(got)) :
        return {
            "ok" : ok,
            "max_diff" : max_diff,
            "max_ratio" : float("nan"),
            "n_violations" : None,
            "worst_index" : None,
        }

    rtol = 1e-5 if rtol is None else rtol
    atol = 1e-5 if atol is None else atol
    threshold = atol + rtol * np.abs(got)
    ratio = np.abs(ref - got) / threshold
    flat = int(np.argmax(ratio))
    worst_index = tuple(int(x) for x in np.unravel_index(flat, ref.shape))
    return {
        "ok" : ok,
        "max_diff" : max_diff,
        "max_ratio" : float(np.max(ratio)),
        "n_violations" : int(np.count_nonzero(np.abs(ref - got) > threshold)),
        "worst_index" : worst_index,
    }