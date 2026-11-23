"""并发极限测试：验证 asyncio.Semaphore(1) 是否能稳定把显卡并发限制为 1。

模拟 10 个不同尺寸矩阵的 GPU 内核执行（不同耗时），统计任意时刻同时占用 GPU 的任务数。
分别跑「有 Semaphore(1)」和「无信号量（对照组）」两种模式，对比并发峰值。
"""

import asyncio
import time

# 与 async_eval.py 保持一致：同时最多 1 个任务占用 GPU
gpu_sem = asyncio.Semaphore(1)

# 10 个不同尺寸（从小到大，模拟不同内核耗时）
MATRIX_SIZES = [128, 192, 256, 320, 384, 448, 512, 768, 1024, 1280]

# 全局并发观测器（单线程事件循环，await 之间不会被抢占，无需加锁）
_active = 0
_max_active = 0


async def _gpu_kernel(matrix_size, sem=None):
    """模拟一次 GPU 内核执行：尺寸越大耗时越长，并记录并发峰值。"""
    global _active, _max_active

    async def _run():
        global _active, _max_active
        _active += 1
        _max_active = max(_max_active, _active)
        # 模拟 kernel 耗时：按尺寸缩放（128 -> ~0.05s，1280 -> ~0.5s）
        await asyncio.sleep(0.05 + matrix_size / 1280 * 0.45)
        _active -= 1
        return matrix_size

    if sem is None:
        return await _run()
    async with sem:
        return await _run()


async def _round(sizes, sem):
    """并发跑一轮，返回 (并发峰值, 总耗时)。"""
    global _active, _max_active
    _active = 0
    _max_active = 0
    start = time.perf_counter()
    await asyncio.gather(*(_gpu_kernel(s, sem) for s in sizes))
    return _max_active, time.perf_counter() - start


async def main():
    # 对照组：无信号量，理应并发峰值 > 1
    max_no_sem, dur_no_sem = await _round(MATRIX_SIZES, None)
    # 实验组：Semaphore(1)，期望并发峰值 == 1
    max_sem, dur_sem = await _round(MATRIX_SIZES, gpu_sem)

    print(f"矩阵尺寸数: {len(MATRIX_SIZES)}，尺寸: {MATRIX_SIZES}")
    print(f"[对照组] 无信号量：并发峰值 = {max_no_sem}，总耗时 = {dur_no_sem:.3f}s")
    print(f"[实验组] Semaphore(1)：并发峰值 = {max_sem}，总耗时 = {dur_sem:.3f}s")

    # 结论
    if max_sem == 1:
        print("\n结论：✅ Semaphore(1) 稳定把显卡并发限制为 1")
    else:
        print(f"\n结论：❌ Semaphore(1) 失效，实测峰值并发 = {max_sem}")


if __name__ == "__main__":
    asyncio.run(main())
