🚀 AI Infra 算子调优系统：长线演进蓝图

🟢 阶段一：工程化与闭环完善

此阶段重点在于补齐系统短板，让 Demo 具备工业级稳定性和通用性。

1. 多模型与多尺寸压测
   · 完善：接入 DeepSeek、通义千问、Kimi 等多个模型 API，测试 256, 512, 1024, 2048 等不同尺寸的矩阵。
   · 目的：积累至少 100 组数据，验证系统在不同场景下的鲁棒性，为写论文提供厚实的数据支撑。
2. 真正的“自我纠错”闭环 (Self-Correction)
   · 完善：目前的循环只是简单的重试。进阶为：通过 subprocess 捕获到具体的编译器报错（如 nvcc error 行号），将其自动拼接到新的 Prompt 中喂回大模型。
   · 目的：让系统真正具备“Agent（智能体）”的雏形，不再依赖人工干预。
3. Web 可视化看板 (Streamlit / Gradio)
   · 完善：写一个 50-100 行的 Python 前端。
   · 目的：脱离黑框框，实现网页端输入提示词 -> 点击运行 -> 动态展示折线图和加速比对比。答辩时的绝杀武器。
4. 完善防作弊与哈希校验
   · 完善：在 Python 侧用纯 C 跑出基准结果，生成 MD5 哈希值。将沙箱里 GPU 跑出的结果也转为哈希值，绝对比对。
   · 目的：防止大模型“背诵”答案，确保 100% 的真实性。

---

🟡 阶段二：知识增强与底层深挖

此阶段开始触及 AI Infra 的核心，涉足真正的性能优化。

5. 引入 RAG (检索增强生成) 知识库
   · 完善：将 NVIDIA 官方 CUDA 优化手册（Best Practices Guide）、CUTLASS 库的优秀代码切片，存入向量数据库（如 ChromaDB）。
   · 动作：大模型写代码前，先根据需求检索相关"最佳实践"文档，作为 Context 喂给它。
   · 目的：极大降低大模型的"幻觉"，让它写出带 __shared__ 或 Tiling 的高级优化代码，而不是简单的暴力循环。
   · ✅ 初步实现（2026-11-18，对应《RAG知识库设计与实施计划》R0/R1/R2）：
     - knowledge/strategies/*.json：6 张 matmul 策略卡（tiling / split-K / vectorized load /
       register tiling / double buffering / occupancy），用 JSON 零新依赖（暂不引 ChromaDB/PyYAML）。
     - rag.py：零依赖 BM25 + 硬过滤（算子族 + 瓶颈指纹 + 语义约束）+ 置信度/近因加权；纯函数离线单测。
     - async_eval.py：首轮 + 画像后两处注入；results 记录 knowledge{enabled,mode,top_k,retrieved_card_ids}；
       KNOWLEDGE_ENABLED 消融开关（默认关）。
     - test_rag.py：27 项单测全过。
     ✅ R3 自蒸馏 + 归因（2026-11-18）：rag.infer_applied_cards / compute_confidence_updates /
        generalize_shape；eval_logic.compute_speedup；async_eval 每轮记录 source_features / speedup_vs_prev /
        applied_cards，best 更快则蒸馏候选卡到 knowledge/candidates/（不自动污染 strategies/），置信度增量
        落盘 results/{run_id}/knowledge_updates.json 供人工 review；新增 OCCUPANCY_01 卡（P1-2：latency+naive 不再 0 命中）。
     待做（R4 实验）：失败→failure_modes 更新需 LLM 代码 diff；Recall@3 标注集与 kb on/off 消融数据尚未建。
6. 接入 Profiling 性能分析工具 (Nsight Compute)
   · 完善：学习命令行工具，在宿主侧（WSL2）运行 ncu，抓取底层的显存带宽利用率、L2 Cache 命中率、SM 计算单元占用率。
   · 动作：如果发现“计算单元没吃满，但显存读取很高”，系统自动将“你的代码存在内存瓶颈”作为 Prompt 反馈给 AI。
   · 目的：从“只看总耗时”升级为“基于硬件画像的精准调优”。
   · ⚠️ 代码完成 + 单测全过 + 循环可端到端跑；真实 ncu 指标仍待用户两步环境操作：
     - profiler.py：ncu 命令构造（宿主侧，--clock-control none / --target-processes all /
       --kernel-name-base demangled / --kernel-name 正则）/ CSV 解析 / 瓶颈分类
       memory·compute·latency（三类均生成策略反馈）。
     - roofline.py（新增）：counter-free 兜底画像——实测时间 + 算子规格 + 源码特征
       （__shared__/float4）做 roofline 判据，ncu 不可用时循环仍能产出瓶颈反馈；
       这是「知识可迁移」研究点（计数器指纹跨卡迁移会断，counter-free 不会）。
     - async_eval.py：成功路径优先宿主 ncu，失败降级 roofline proxy；best-so-far 保优；
       API 调用带超时/退避重试，gather 用 return_exceptions 防单尺寸异常毁整批。
     - eval_logic.py（新增）：results schema v2 落盘（results/{run_id}/{N}.json +
       index.jsonl，含 run_id/时间戳/response.model 真实名/环境快照/每轮 code 全文与错误反馈）。
     - validate.py：float64 参考 + 按尺寸校准容差（修复 2048 正确内核被误判）。
     仍需用户做的两件事（代码代不了）：① Windows 控制面板开 GPU 性能计数器；
     ② WSL 从 NVIDIA 官网装支持 GB206 的最新 Nsight Compute，装后
     `ncu --list-chips | grep -i gb206` 应能搜到。
7. 引入 Auto-tuning (自动参数搜索)
   · 完善：结合简单的暴力搜索或遗传算法。
   · 动作：让大模型生成代码骨架，参数部分（如 block_size, grid_size）留空。系统自动遍历测试不同参数组合，找出最优解。
   · 目的：算法与工程结合，榨干 RTX 5060 的最后一滴算力。

---

🔴 阶段三：前沿探索与多智能体

此阶段对标科研级系统，甚至可以直接作为保研/直博的敲门砖。

8. Multi-Agent (多智能体) 协同系统
   · 完善：打造三个独立的 AI 角色。
     · Coder Agent：负责写代码。
     · Reviewer Agent：负责 Review 代码逻辑，检查有无死锁、越界风险。
     · Tester Agent：负责在沙箱中跑测试并提取数据。
   · 目的：让 AI 内部互相博弈、互相纠错，实现无人值守的代码进化。
9. 算子融合 (Operator Fusion) 探索
   · 完善：不再局限于单个矩阵乘法。尝试让 AI 自动推导：能不能把 矩阵乘法 + ReLU激活函数 合并写在同一个 CUDA Kernel 里？
   · 目的：减少显存 I/O 搬运时间，这是当前大模型推理加速的核心瓶颈（如 FlashAttention 的核心思想）。
10. 探索形式化验证 (SMT Solver)
    · 完善：对于 AI 生成的代码，不仅靠测试用例跑分，还尝试用数学逻辑（如 Z3 求解器）证明其逻辑的绝对正确性。
    · 目的：解决 AI 在安全攸关（如自动驾驶）等底层代码生成中的信任问题。


    // 1.测试不同 block size对kernel耗时的影响
    // 2.引入Unified Memory简化分配内存