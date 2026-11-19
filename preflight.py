#!/usr/bin/env python3
"""周末跑批前预检（P0-4）：全部通过才开跑，失败项给出修复命令。

检查项：
  1. .env 里的 DEEPSEEK_API_KEY
  2. GPU 可用（nvidia-smi）
  3. ncu 版本（默认需 2026.4.0）且认识 gb206
  4. Docker 沙箱镜像已缓存
  5. 磁盘剩余空间
  6. 全部离线单测通过

用法：
    python preflight.py                 # 全部检查，任一失败退出码 1
    python preflight.py --ncu-version 2026.4.0 --min-disk-gb 5
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import shutil
import subprocess
import sys

from dotenv import load_dotenv

HERE = pathlib.Path(__file__).resolve().parent


def _run(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
    """同步跑子进程，返回 (returncode, stdout, stderr)。"""
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, text=True)
        return r.returncode, r.stdout or "", r.stderr or ""
    except subprocess.TimeoutExpired:
        return 1, "", f"超时（>{timeout}s）"
    except FileNotFoundError:
        return 1, "", f"命令不存在：{args[0]}"
    except Exception as e:
        return 1, "", f"异常：{e}"


def check_dotenv() -> tuple[bool, str, str]:
    key = os.getenv("DEEPSEEK_API_KEY") or ""
    if key.strip():
        return True, f"DEEPSEEK_API_KEY 已设置（{key[:8]}…{key[-4:]}）", ""
    return False, "DEEPSEEK_API_KEY 缺失或为空", "在项目根 .env 里设置 DEEPSEEK_API_KEY=sk-xxx"


def check_gpu() -> tuple[bool, str, str]:
    rc, out, err = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if rc == 0 and out.strip():
        return True, f"GPU 可用：{out.strip().splitlines()[0]}", ""
    return False, "nvidia-smi 不可用（无 GPU 或驱动异常）", f"检查驱动：{err.strip()[-200:]}"


def check_ncu(expected_version: str) -> tuple[bool, str, str]:
    try:
        import async_eval
    except Exception as e:
        return False, f"无法导入 async_eval（{e}）", "确认依赖已安装（numpy/openai/python-dotenv）"
    ncu_bin = async_eval.get_ncu_bin()
    rc, out, err = _run([ncu_bin, "--version"])
    ver_ok = expected_version in (out + err)
    gb206_ok = async_eval.ncu_supports_gb206()
    if ver_ok and gb206_ok:
        return True, f"ncu {expected_version} 且认识 gb206（{ncu_bin}）", ""
    problems = []
    if not ver_ok:
        problems.append(f"版本不含 {expected_version}（当前 {ncu_bin}）")
    if not gb206_ok:
        problems.append("不认识 gb206（--list-chips 无 gb206）")
    return False, "ncu 不可用：" + "；".join(problems), (
        f"升级 ncu 到 {expected_version} 并加入 PATH（参考 README/笔记：新版 ncu 常装在 "
        "~/.local/nsight-compute/*/ncu）")


def check_docker_image() -> tuple[bool, str, str]:
    try:
        import async_eval
        image = async_eval.CUDA_IMAGE
    except Exception as e:
        return False, f"无法导入 async_eval（{e}）", "确认依赖已安装"
    rc, out, _ = _run(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    if rc == 0 and out.strip():
        return True, f"沙箱镜像已缓存：{image}", ""
    return False, f"沙箱镜像未缓存：{image}", f"docker pull {image}"


def check_disk(min_gb: int) -> tuple[bool, str, str]:
    try:
        free = shutil.disk_usage(HERE).free / (1024 ** 3)
    except OSError as e:
        return False, f"无法读取磁盘空间（{e}）", "检查项目所在磁盘挂载"
    if free >= min_gb:
        return True, f"磁盘剩余 {free:.1f} GB（≥{min_gb} GB）", ""
    return False, f"磁盘剩余 {free:.1f} GB（<{min_gb} GB）", "清理磁盘或换盘；results/ 与 experiments/ 可打包归档后删除"


def _run_unit_tests() -> tuple[int, int, list[str]]:
    """轻量测试运行器：只导入含 test_* 函数的 test_*.py，逐个跑（不依赖 pytest）。"""
    passed = failed = 0
    failures: list[str] = []
    sys.path.insert(0, str(HERE))
    for p in sorted(HERE.glob("test_*.py")):
        text = p.read_text(encoding="utf-8")
        if "def test_" not in text:
            continue
        try:
            mod = importlib.import_module(p.stem)
        except Exception as e:
            failed += 1
            failures.append(f"{p.stem}: import 失败：{e!r}")
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception as e:
                failed += 1
                failures.append(f"{p.stem}.{name}: {e!r}")
    return passed, failed, failures


def check_tests() -> tuple[bool, str, str]:
    passed, failed, failures = _run_unit_tests()
    if failed == 0:
        return True, f"全部单测通过（{passed} 项）", ""
    detail = "；".join(failures[:5])
    return False, f"单测 {passed} 过 / {failed} 失败：{detail}", "运行 python preflight.py 查看全部失败项"


def main() -> int:
    ap = argparse.ArgumentParser(description="周末跑批前预检（P0-4）")
    ap.add_argument("--ncu-version", default="2026.4.0", help="要求的 ncu 版本（默认 2026.4.0）")
    ap.add_argument("--min-disk-gb", type=float, default=5.0, help="最小磁盘剩余 GB（默认 5）")
    args = ap.parse_args()

    load_dotenv(os.path.join(HERE, ".env"))

    checks = [
        (".env 密钥", check_dotenv()),
        ("GPU", check_gpu()),
        ("ncu", check_ncu(args.ncu_version)),
        ("Docker 镜像", check_docker_image()),
        ("磁盘", check_disk(args.min_disk_gb)),
        ("单测", check_tests()),
    ]

    all_ok = True
    print("========== 跑批预检 ==========")
    for name, (ok, msg, fix) in checks:
        icon = "✅" if ok else "❌"
        print(f"{icon} {name}：{msg}")
        if not ok and fix:
            print(f"    ↳ 修复：{fix}")
        all_ok = all_ok and ok

    print("==============================")
    if all_ok:
        print("✅ 全绿，可以开跑。")
        return 0
    print("❌ 存在未通过项，先修复再开跑。")
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
