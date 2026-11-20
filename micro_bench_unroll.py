#!/usr/bin/env python3
"""微基准：循环展开（loop unrolling）在 RTX 5060 Laptop / sm_120 上的真实效果。

不依赖 LLM / API，只做「固定 matmul 内核 × unroll 因子」的受控实验：
  - 内核族：naive（无 tiling）、tiled1（每线程 1 输出）、tiled2（每线程 2x2 寄存器分块）
  - 维度：TILE ∈ {16, 32}（tiled 族）、unroll ∈ {1, 2, 4, 8, full}
  - 编译：docker nvidia/cuda:12.8.2 + nvcc -O3 -arch=sm_120 -Xptxas -v（记录寄存器/spill）
  - 测量：宿主侧 ncu（2026.4.0，支持 gb206）采 kernel 耗时/寄存器/指令数/占用率；
          失败时回退到二进制自带 cudaEvent 计时（5 次中位数）
  - 正确性：每个变体输出矩阵与 float64 参考比对（校准容差）

用法：
    python micro_bench_unroll.py                    # 跑全部变体
    python micro_bench_unroll.py --only tiled1_16_4 # 只跑一个变体（调试）
    python micro_bench_unroll.py --n 1024 --skip-ncu

输出：experiments/micro_unroll/{src,bin,logs}/ + results.csv + summary 打印。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "experiments", "micro_unroll")
IMAGE = "nvidia/cuda:12.8.2-devel-ubuntu22.04"

NCU_METRICS = [
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "smsp__inst_executed.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

# ---------------------------------------------------------------------------
# CUDA 源码模板
# ---------------------------------------------------------------------------

MAIN_TMPL = r"""
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define N __N__
#define TILE __TILE__

__KERNEL__

int main() {
    const size_t bytes = (size_t)N * N * sizeof(float);
    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            hA[i * N + j] = ((i + j) % 10) * 0.1f;
            hB[i * N + j] = ((i * 3 + j) % 10) * 0.1f;
        }
    }
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(__BLOCK__);
    dim3 grid(__GRID__);

    // warmup 3 次
    for (int i = 0; i < 3; ++i) matmul<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaEvent_t st, en;
    cudaEventCreate(&st);
    cudaEventCreate(&en);
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(st);
        matmul<<<grid, block>>>(dA, dB, dC);
        cudaEventRecord(en);
        cudaEventSynchronize(en);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, st, en);
        printf("Time: %.9f seconds\n", ms / 1000.0);
    }

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    printf("C[0][0] = %.6f\n", hC[0]);
    FILE* fp = fopen("cuda_output___N__.bin", "wb");
    if (fp) {
        fwrite(hC, sizeof(float), (size_t)N * N, fp);
        fclose(fp);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
"""

NAIVE = r"""
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
__UNROLL_PRAGMA__
    for (int k = 0; k < N; ++k) sum = fmaf(A[row * N + k], B[k * N + col], sum);
    C[row * N + col] = sum;
}
"""

TILED1 = r"""
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
__UNROLL_PRAGMA__
        for (int k = 0; k < TILE; ++k)
            sum = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
        __syncthreads();
    }
    C[row * N + col] = sum;
}
"""

TILED2 = r"""
__global__ void matmul(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x;   // 0..TILE/2-1
    const int ty = threadIdx.y;   // 0..TILE/2-1
    const int bm = blockIdx.y * TILE;
    const int bn = blockIdx.x * TILE;
    const int tid = ty * (TILE / 2) + tx;
    const int nthreads = (TILE / 2) * (TILE / 2);
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        for (int idx = tid; idx < TILE * TILE; idx += nthreads) {
            int r = idx / TILE;
            int c = idx % TILE;
            As[r][c] = A[(bm + r) * N + t * TILE + c];
            Bs[r][c] = B[(t * TILE + r) * N + bn + c];
        }
        __syncthreads();
__UNROLL_PRAGMA__
        for (int k = 0; k < TILE; ++k) {
            float a0 = As[ty * 2 + 0][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2 + 0];
            float b1 = Bs[k][tx * 2 + 1];
            c00 = fmaf(a0, b0, c00);
            c01 = fmaf(a0, b1, c01);
            c10 = fmaf(a1, b0, c10);
            c11 = fmaf(a1, b1, c11);
        }
        __syncthreads();
    }
    C[(bm + ty * 2 + 0) * N + bn + tx * 2 + 0] = c00;
    C[(bm + ty * 2 + 0) * N + bn + tx * 2 + 1] = c01;
    C[(bm + ty * 2 + 1) * N + bn + tx * 2 + 0] = c10;
    C[(bm + ty * 2 + 1) * N + bn + tx * 2 + 1] = c11;
}
"""

KERNELS = {"naive": NAIVE, "tiled1": TILED1, "tiled2": TILED2}


def unroll_pragma(u: str) -> str:
    if u == "full":
        return "#pragma unroll"
    return f"#pragma unroll {u}"


def block_grid(family: str, tile: int, n: int) -> tuple[str, str, int]:
    """返回 (block 表达式, grid 表达式, 线程数)。"""
    if family == "naive":
        return "dim3(16, 16)", "dim3((N + 15) / 16, (N + 15) / 16)", 256
    if family == "tiled1":
        return f"dim3({tile}, {tile})", f"dim3(N / {tile}, N / {tile})", tile * tile
    if family == "tiled2":
        return f"dim3({tile // 2}, {tile // 2})", f"dim3(N / {tile}, N / {tile})", (tile // 2) ** 2
    raise ValueError(family)


def build_variants(n: int) -> list[dict]:
    variants = []
    # naive：unroll 1 / 4 / full（看无 tiling 时展开有没有用）
    for u in ("1", "4", "full"):
        variants.append({"name": f"naive_u{u}", "family": "naive", "tile": 16, "unroll": u})
    for family in ("tiled1", "tiled2"):
        for tile in (16, 32):
            for u in ("1", "2", "4", "8", "full"):
                variants.append({"name": f"{family}_{tile}_u{u}", "family": family, "tile": tile, "unroll": u})
    return variants


def gen_source(v: dict, n: int) -> str:
    family, tile, u = v["family"], v["tile"], v["unroll"]
    body = KERNELS[family].replace("__UNROLL_PRAGMA__", unroll_pragma(u))
    block_expr, grid_expr, _ = block_grid(family, tile, n)
    src = (MAIN_TMPL
           .replace("__KERNEL__", body)
           .replace("__N__", str(n))
           .replace("__TILE__", str(tile))
           .replace("__BLOCK__", block_expr)
           .replace("__GRID__", grid_expr))
    return src


# ---------------------------------------------------------------------------
# 编译 / 测量
# ---------------------------------------------------------------------------

def find_ncu() -> str | None:
    """找支持 gb206 的 ncu（新版常在 ~/.local/nsight-compute/*/ncu）。"""
    import glob
    cands = []
    env_bin = os.getenv("NCU_BIN")
    if env_bin:
        cands.append(env_bin)
    home = os.path.expanduser("~")
    cands += sorted(glob.glob(os.path.join(home, ".local", "nsight-compute", "*", "ncu")), reverse=True)
    which = shutil.which("ncu")
    if which:
        cands.append(which)
    for c in cands:
        try:
            out = subprocess.run([c, "--list-chips"], capture_output=True, text=True, timeout=15)
            if "gb206" in ((out.stdout or "") + (out.stderr or "")).lower():
                return c
        except Exception:
            continue
    return None


def compile_variant(v: dict) -> tuple[bool, str, str]:
    """容器内编译；返回 (ok, 合并日志, 错误)。"""
    name = v["name"]
    src_rel = f"src/{name}.cu"
    bin_rel = f"bin/{name}"
    log_path = os.path.join(WORK, "logs", f"{name}.log")
    # 注意：不在容器里用 tee 写日志（容器 root 写出的文件宿主用户无法覆盖）；日志由宿主写。
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{WORK}:/app", IMAGE,
        "bash", "-c",
        f"cd /app && nvcc -O3 -arch=sm_120 -Xptxas -v -o {bin_rel} {src_rel}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False, "", "compile timeout"
    log = (r.stdout or "") + (r.stderr or "")
    try:
        if os.path.exists(log_path):
            os.remove(log_path)
    except OSError:
        pass
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log)
    ok = os.path.isfile(os.path.join(WORK, bin_rel)) and r.returncode == 0
    return ok, log, "" if ok else log[-300:]


def parse_ptxas(log: str) -> dict:
    reg = re.search(r"Used (\d+) registers", log)
    spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", log)
    return {
        "registers": int(reg.group(1)) if reg else None,
        "spill_stores": int(spill.group(1)) if spill else 0,
        "spill_loads": int(spill.group(2)) if spill else 0,
    }


def parse_ncu_csv(text: str) -> dict:
    """解析 ncu --csv 输出（表头驱动，跳过 app 自身 stdout 噪声）。"""
    import csv as _csv
    import io as _io
    rows = list(_csv.reader(_io.StringIO(text)))
    name_idx = val_idx = None
    out = {}
    for row in rows:
        if not row:
            continue
        lowered = [c.strip().lower() for c in row]
        if name_idx is None:
            if "metric name" in lowered and "metric value" in lowered:
                name_idx = lowered.index("metric name")
                val_idx = lowered.index("metric value")
            continue
        if len(row) <= max(name_idx, val_idx):
            continue
        metric = row[name_idx].strip()
        if metric in NCU_METRICS:
            try:
                out[metric] = float(row[val_idx])
            except ValueError:
                pass
    return out


def run_ncu(ncu_bin: str, v: dict) -> tuple[dict, str]:
    """宿主 ncu profile；返回 (metrics, 错误)。"""
    cmd = [
        ncu_bin, "--csv", "--print-metric-name", "name",
        "--clock-control", "none", "--target-processes", "all",
        "--kernel-name-base", "demangled", "--kernel-name", "regex:matmul",
        "--launch-skip", "3", "--launch-count", "1",
        "--metrics", ",".join(NCU_METRICS),
        f"./bin/{v['name']}",
    ]
    try:
        r = subprocess.run(cmd, cwd=WORK, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {}, "ncu timeout"
    metrics = parse_ncu_csv(r.stdout or "")
    if not metrics:
        return {}, ((r.stderr or r.stdout or "").strip()[-300:] or "no metrics")
    return metrics, ""


def run_selftime(v: dict) -> tuple[float | None, str]:
    """直接跑二进制，取 5 次 cudaEvent 计时的中位数（ncu 不可用时回退）。"""
    try:
        r = subprocess.run([f"./bin/{v['name']}"], cwd=WORK, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "run timeout"
    vals = [float(m.group(1)) for m in re.finditer(r"Time:\s*([0-9.eE+-]+) seconds", r.stdout or "")]
    if not vals:
        return None, ((r.stderr or r.stdout or "").strip()[-200:] or "no Time")
    vals.sort()
    return vals[len(vals) // 2], ""


def check_correctness(n: int, work: str) -> tuple[bool, float | None, str]:
    """用仓库 validate.py 的 float64 参考 + 校准容差检查输出矩阵。"""
    sys.path.insert(0, HERE)
    import validate
    path = os.path.join(work, f"cuda_output_{n}.bin")
    if not os.path.isfile(path):
        return False, None, "missing output bin"
    try:
        ref = validate.reference_matmul(n)
        got = validate.load_matrix(path, n)
        rtol, atol = validate.default_tolerance(n)
        ok, diff = validate.matrices_close(ref, got, rtol=rtol, atol=atol)
        return bool(ok), float(diff), ""
    except Exception as e:
        return False, None, f"validate error: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="循环展开微基准（sm_120）")
    ap.add_argument("--n", type=int, default=1024, help="矩阵尺寸（默认 1024）")
    ap.add_argument("--only", default=None, help="只跑名字包含该子串的变体（调试）")
    ap.add_argument("--skip-ncu", action="store_true", help="不用 ncu，只用自带 cudaEvent 计时")
    args = ap.parse_args()

    for sub in ("src", "bin", "logs"):
        os.makedirs(os.path.join(WORK, sub), exist_ok=True)

    variants = build_variants(args.n)
    if args.only:
        variants = [v for v in variants if args.only in v["name"]]

    ncu_bin = None if args.skip_ncu else find_ncu()
    print(f"变体数: {len(variants)} | ncu: {ncu_bin or '不可用（回退自带计时）'}")

    rows = []
    for i, v in enumerate(variants, 1):
        src = gen_source(v, args.n)
        src_path = os.path.join(WORK, "src", f"{v['name']}.cu")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src)

        ok, log, err = compile_variant(v)
        row = {
            "name": v["name"], "family": v["family"], "tile": v["tile"], "unroll": v["unroll"],
            "n": args.n, "compile_ok": ok, **parse_ptxas(log), "ncu_time_sec": None,
            "self_time_sec": None, "registers_ncu": None, "inst_executed": None,
            "sm_pct": None, "dram_pct": None, "occ_pct": None,
            "correct": None, "max_diff": None, "error": err,
        }
        if not ok:
            print(f"[{i}/{len(variants)}] {v['name']:<22} 编译失败: {err[-120:]}")
            rows.append(row)
            continue

        if ncu_bin:
            metrics, nerr = run_ncu(ncu_bin, v)
            if metrics:
                row["ncu_time_sec"] = metrics.get("gpu__time_duration.sum", 0) / 1e9 if metrics.get("gpu__time_duration.sum") else None
                row["registers_ncu"] = metrics.get("launch__registers_per_thread")
                row["inst_executed"] = metrics.get("smsp__inst_executed.sum")
                row["sm_pct"] = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed")
                row["dram_pct"] = metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
                row["occ_pct"] = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active")
            elif nerr:
                row["error"] = f"ncu: {nerr}"

        if row["ncu_time_sec"] is None:
            t, serr = run_selftime(v)
            row["self_time_sec"] = t
            if serr and not row["error"]:
                row["error"] = f"self-time: {serr}"

        correct, diff, cerr = check_correctness(args.n, WORK)
        row["correct"] = correct
        row["max_diff"] = diff
        if cerr and not row["error"]:
            row["error"] = cerr

        print(f"[{i}/{len(variants)}] {v['name']:<22} "
              f"ncu={row['ncu_time_sec'] if row['ncu_time_sec'] is None else round(row['ncu_time_sec']*1e3,3)}ms "
              f"self={row['self_time_sec'] if row['self_time_sec'] is None else round(row['self_time_sec']*1e3,3)}ms "
              f"reg={row['registers']} spill={row['spill_stores']}/{row['spill_loads']} "
              f"correct={row['correct']} err={row['error'][:60]}")
        rows.append(row)

    # 汇总
    out_csv = os.path.join(WORK, "results.csv")
    fieldnames = ["name", "family", "tile", "unroll", "n", "compile_ok", "registers",
                  "spill_stores", "spill_loads", "ncu_time_sec", "self_time_sec",
                  "registers_ncu", "inst_executed", "sm_pct", "dram_pct", "occ_pct",
                  "correct", "max_diff", "error"]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print("\n========== 汇总（按 family/tile 内相对 unroll=1 的加速比） ==========")
    print(f"{'variant':<22}{'time(ms)':>10}{'speedup':>9}{'reg':>6}{'spill':>8}{'inst(M)':>9}{'sm%':>7}{'dram%':>7}{'ok':>4}")
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["family"], r["tile"]), []).append(r)
    for key, items in sorted(groups.items()):
        base = next((x for x in items if x["unroll"] == "1"), None)
        base_t = (base or {}).get("ncu_time_sec") or (base or {}).get("self_time_sec")
        for r in items:
            t = r["ncu_time_sec"] or r["self_time_sec"]
            sp = (base_t / t) if (base_t and t) else None
            inst = (r["inst_executed"] / 1e6) if r["inst_executed"] else None
            spill = f"{r['spill_stores']}/{r['spill_loads']}" if (r["spill_stores"] or r["spill_loads"]) else "-"
            print(f"{r['name']:<22}{(t*1e3 if t else 0):>10.3f}{(sp if sp else 0):>9.2f}"
                  f"{(r['registers'] or 0):>6}{spill:>8}{(inst if inst else 0):>9.1f}"
                  f"{(r['sm_pct'] or 0):>7.1f}{(r['dram_pct'] or 0):>7.1f}{str(r['correct']):>4}")
    print(f"\n结果 CSV: {out_csv}")
    print(f"源码/二进制/日志: {WORK}/{{src,bin,logs}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
