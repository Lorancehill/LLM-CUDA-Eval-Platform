"""validate.py 单测：float64 参考 + 按尺寸校准容差（B5：2048 误判回归）。"""

import numpy as np

import validate


def test_reference_matmul_is_float64():
    assert validate.reference_matmul(64).dtype == np.float64


def test_default_tolerance_grows_with_N():
    rtol_small, _ = validate.default_tolerance(256)
    rtol_large, _ = validate.default_tolerance(2048)
    assert rtol_large > rtol_small


def test_correct_float32_kernel_passes_at_2048():
    """模拟顺序累加 float32 内核在 2048 的舍入误差，应通过校准阈值。

    B5 回归：旧实现（float32 参考 + 固定 1e-5）会把 2048 的正确内核误判为错误
    （实测 max_diff ~5e-3 > 阈值 ~4.9e-3）。float64 参考 + rtol∝N 后应通过。
    """
    N = 2048
    ref = validate.reference_matmul(N)  # float64
    A, B = validate.init_matrices_float64(N)
    # 用 float32 matmul 作为「正确的 float32 内核」的代理（真实内核用不同累加顺序，
    # 但相对误差同量级，都远小于 rtol=2*N*eps32 的上界）。
    got = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    ok, max_diff = validate.matrices_close(ref, got, *validate.default_tolerance(N))
    assert ok, f"正确 float32 内核被误判（max_diff={max_diff:.3g}）"


def test_wrong_kernel_rejected_at_2048():
    N = 2048
    ref = validate.reference_matmul(N)
    A, B = validate.init_matrices_float64(N)
    got = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float32)
    got[0, 0] += 10.0  # 单点大错误，远大于任何合理舍入误差
    ok, _ = validate.matrices_close(ref, got, *validate.default_tolerance(N))
    assert not ok


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
