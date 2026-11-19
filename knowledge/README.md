# 知识库（Strategy Cards）生命周期说明

本目录承载 RAG 的「策略卡」知识资产，结构：

```
knowledge/
├── README.md          # 本文件
├── docs/              # 卡片 provenance 引用的文档语料（论文审计用，暂空）
├── strategies/        # 已入库、供检索注入的策略卡（唯一真相来源，进版本库）
└── candidates/        # 自蒸馏候选卡（gitignore，人工 review 后晋升，勿自动污染 strategies/）
```

## 1. 卡片状态（curated → validated → proven）

| 状态 | 含义 | 依据 |
|---|---|---|
| `curated` | 文档/经验蒸馏而来，未经本项目实验验证 | provenance 指向文档，confidence 初值 ≤0.6 |
| `validated` | ≥2 次独立实验支持（evidence 里 ≥2 条不同 run） | 人工 review evidence 后手动改状态 |
| `proven` | 跨尺寸/跨算子/跨卡验证 | 人工 review 后手动改状态 |

状态**只由人工 review 变更**；自动管线只改 `confidence` 与追加 `evidence`，不改状态。

## 2. 知识如何增长（R3 自蒸馏闭环）

1. 评测运行（`async_eval.py`）落盘：
   - `results/{run_id}/{N}.json`：`knowledge.knowledge_updates_applied`（每卡的 delta/rounds/evidence）与 `candidates_created`；
   - `results/{run_id}/knowledge_updates_{N}.json`：同名待合并增量（文件名带尺寸，避免并发覆盖）；
   - `knowledge/candidates/{SELF_...-{N}-a{attempt}}.json`：模型用**知识库未覆盖的策略**提速时蒸馏的新候选卡（id 含尺寸与轮次）。
2. 人工 review：
   - 查看待合并增量：`python rag.py updates`
   - 合并增量/证据回策略卡（并消费更新文件）：`python rag.py merge`（可加 `--dry-run` 先预览）
   - 把候选卡晋升进 strategies/：`python rag.py promote`（可加 `--dry-run`）

## 3. 合并/晋升命令

```bash
python rag.py updates                      # 查看待合并的置信度增量与证据
python rag.py merge --dry-run              # 预览合并结果，不落盘
python rag.py merge                        # 写回 confidence + evidence，删除已消费的更新文件
python rag.py promote --dry-run            # 预览候选卡晋升，不落盘
python rag.py promote                      # 候选卡 status → curated 并移入 strategies/
```

`merge` 只把 `delta` 累加到卡片 `confidence`（用 `clamp_confidence` 夹取到 [0.1, 0.95]）并追加 `evidence`；
不会改 `status`，不会覆盖卡片其它字段。

## 4. 消融与可信度约定

- 知识库开关：`KNOWLEDGE_ENABLED=1/0`；`KNOWLEDGE_TOPK` 控制每轮注入条数。
- 论文口径（见《论文可行性评估与补实验清单》3.1）：
  - 计时以 ncu 可信计时 `ncu_time_sec`（kernel-only）为准，自报 `Time:` 仅作对照；
  - 每个内核计时跑 `TIMING_REPEATS`（默认 5）次取中位数；
  - 环境快照记录完整 ncu 版本行与 GPU 温度/功耗。
- `knowledge/candidates/` 不进版本库；论文需要时归档候选卡到 `results/` 或单独数据目录。
