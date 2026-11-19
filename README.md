# LLM-CUDA-Eval-Platform

让大模型自动生成、评测并迭代优化 CUDA 矩阵乘法内核的闭环平台。目标：验证「LLM + 硬件画像（Nsight Compute）+ RAG 知识库」能否稳定产出超越 naive 实现、甚至接近手写优化的算子代码。

当前阶段：单机 RTX 5060（sm_120）上跑通生成 → 编译 → 运行 → 校验 → 画像 → 反馈 → 再生成的自动迭代闭环，配套批跑 / 预检 / 看板 / 微基准 / 知识库（自蒸馏）。

## 核心闭环

1. **生成**：`async_eval.py` 调 DeepSeek API，要求模型输出完整 CUDA 代码。
2. **沙箱评测**：`docker run`（`nvidia/cuda:12.8.2-devel-ubuntu22.04`）内 `nvcc -arch=sm_120` 编译并跑 `TIMING_REPEATS` 次取中位数。
3. **校验**：`validate.py` 与 float64 参考矩阵做按尺寸校准的容差比对。
4. **画像**：优先宿主 `ncu`（kernel-only 可信计时 + 硬件指标），失败降级 `roofline.py` 的 counter-free 指纹。
5. **反馈**：`eval_logic.py` 结构化编译错误（行号 + 出错行源码）/ 瓶颈反馈喂回模型继续迭代。
6. **知识库**：`rag.py` 策略卡检索注入 + 自蒸馏候选卡 + 置信度更新（人工 review 后合并）。

## 快速开始

```bash
# 环境
pip install -r requirements.txt
cp .env.example .env   # 或直接写 .env：DEEPSEEK_API_KEY=sk-xxx

# 预检（密钥 / GPU / ncu / Docker / 磁盘 / 单测）
python preflight.py

# 单次评测（默认 256,512,1024 三尺寸）
python async_eval.py

# 正式实验（kb on/off × 3 重复，带实验标签，推荐先 dry-run 验证链路）
python run_experiment.py --exp exp1 --sizes 256,512,1024 --kb 0,1 --repeats 3 --dry-run
python run_experiment.py --exp exp1 --sizes 256,512,1024 --kb 0,1 --repeats 3

# 可视化看板
python dashboard.py            # http://127.0.0.1:8000

# 知识库合并（人工 review 后把增量写回策略卡）
python rag.py updates
python rag.py merge --dry-run
python rag.py merge
python rag.py promote --dry-run
```

## 目录结构

| 路径 | 作用 |
|---|---|
| `async_eval.py` | 自动评测主循环（生成/沙箱/校验/画像/反馈） |
| `eval_logic.py` | 纯逻辑：耗时解析、best-so-far、结构化报错、反馈与落盘 schema |
| `validate.py` | 结果矩阵与 float64 参考比对（按尺寸校准容差） |
| `profiler.py` | ncu 画像解析（kernel-only 计时 + SM/DRAM/占用率/L2） |
| `roofline.py` | counter-free roofline 兜底瓶颈指纹 |
| `rag.py` | 策略卡加载/检索/注入/自蒸馏/合并 CLI |
| `dashboard.py` | 零依赖网页看板 |
| `run_experiment.py` | 批跑脚本（注入实验标签，归档 logs/runs/summary.csv） |
| `preflight.py` | 环境预检 |
| `micro_bench_unroll.py` | loop-unrolling 微基准 |
| `knowledge/strategies/*.json` | 策略卡（curated） |
| `knowledge/candidates/` | 自蒸馏候选卡（gitignore，人工 review） |
| `results/` | 证据链落盘（gitignore） |
| `experiments/` | 批跑归档（gitignore） |

## 测试

```bash
for f in test_*.py; do python "$f"; done   # 80+ 项单测，全部离线
python -m compileall .
```

## 安全说明

- `.env` 已 gitignore 且权限 600；仓库内无真实密钥。
- 沙箱容器默认 rw + root（`-v $(pwd):/app`），收集数据阶段保持现状；数据冻结后按报告 P1 #8 加固（`--read-only` + tmpfs 构建目录 + 非 root `--user`）。

## 文档

- `ROADMAP.md`：技术路线与阶段计划
- `NOTES.md`：科研愿景（MVP → 顶会）
- 设计与评审记录：见 `ROADMAP.md` 与 git 提交历史；如需归档，可把评估/评审文档放进 `改进反馈/` 目录
