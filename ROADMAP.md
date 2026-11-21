🚀 AI Infra 算子调优系统：长线演进蓝图

> 核心研究命题：能否让「LLM 写结构 + 搜索器调参数」两级搜索，稳定逼近手写优化内核。
> 本文档在既有三阶段之上，新增一条横向主线 —— **参数级自动搜索（Auto-tuning）算法谱系**：
> 从网格/随机基线，到爬山（Hill Climbing）、坐标下降、模拟退火、遗传算法，再到贝叶斯优化（TPE/GP）
> 与 Hyperband 预算感知搜索，最后用 LLM 热启动把「改代码」与「调参数」两个闭环焊在一起。
> 每级算法都在同一参数空间、同一成本函数下可复现对比，产出可写论文的收敛曲线与消融表。

---

🟢 阶段一：工程化与闭环完善

此阶段重点在于补齐系统短板，让 Demo 具备工业级稳定性和通用性。

1. 多模型与多尺寸压测
   · 完善：接入 DeepSeek、通义千问、Kimi 等多个模型 API，测试 256, 512, 1024, 2048 等不同尺寸的矩阵。
   · 目的：积累至少 100 组数据，验证系统在不同场景下的鲁棒性，为写论文提供厚实的数据支撑。
   · ✅ 周末跑批基建（2026-11-19，P0-3/P0-4 + P1/P2 修复）：
     - run_experiment.py：一键跑批——kb×repeat 矩阵展开成独立子进程，经 RUN_ID/EXPERIMENT_ID/CONDITION/
       REPEAT/KNOWLEDGE_ENABLED/SIZES 环境变量注入；结果归档 experiments/{exp}/runs/、日志 logs/、汇总 summary.csv；
       失败自动重试 1 次，连败 2 次标记 skipped 不中断整批。
     - preflight.py：开跑前预检（.env 密钥 / GPU / ncu / Docker 镜像 / 磁盘 / 全量离线单测），全部通过才开跑。
     - P1 修复：--sizes 经 SIZES 环境变量真正传到 async_eval（原写死 256/512/1024，smoke/分段实验会数据不一致）。
     - P2 修复：失败 run 隔离到 experiments/{exp}/failed/ 并剔除 results/index.jsonl，避免重试残留污染对比；
       summary.csv 每尺寸补 timing_source / cards_loaded / applied_cards / candidates / 首轮 vs 最优，供论文表直出。
2. 真正的“自我纠错”闭环 (Self-Correction)
   · 完善：目前的循环只是简单的重试。进阶为：通过 subprocess 捕获到具体的编译器报错（如 nvcc error 行号），将其自动拼接到新的 Prompt 中喂回大模型。
   · 目的：让系统真正具备“Agent（智能体）”的雏形，不再依赖人工干预。
   · ✅ 已实现（2026-11-19，不再只给 stderr 尾部 300 字符）：
     - eval_logic.extract_compile_errors()：从 nvcc / 宿主 g++ / ptxas 三种编译输出里提取
       结构化报错行（file + 行号 + severity + 报错原因），纯函数离线单测覆盖。
     - async_eval._read_source_context()：按第一条带行号的报错行，读回该 .cu 文件的出错行
       附近源码（带行号、>>> 标记出错行），把「具体是哪一行写错了」直接喂给模型。
     - eval_logic.build_error_feedback()：按失败类别（timeout / run_failed / no_output /
       wrong_result）拼装纠错反馈；有结构化编译错误时优先用「行号 + 报错原因 + 出错行源码」，
       否则兜底回落到 stdout/stderr 尾部。results 每轮 record 新增 compile_errors 字段留证。
     - test_eval_logic.py：新增 6 项单测（nvcc/gcc/ptxas 解析、无匹配、反馈优先结构化/兜底、
       wrong_result/timeout 文案），全过。
3. Web 可视化看板 (Streamlit / Gradio)
   · 完善：写一个 50-100 行的 Python 前端。
   · 目的：脱离黑框框，实现网页端输入提示词 -> 点击运行 -> 动态展示折线图和加速比对比。答辩时的绝杀武器。
   · ✅ 已实现（2026-11-19，零新依赖，纯标准库 http.server + 内嵌 HTML/Canvas，不引 Streamlit/Gradio）：
     - dashboard.py：读取 results/index.jsonl + results/{run_id}/{size}.json（schema v2），
       按 run 聚合为 `/api/data` JSON；`python dashboard.py` 即启（默认 http://127.0.0.1:8000）。
     - 看板内容：环境快照卡 + 汇总卡（尺寸/成功率/总轮次/最佳可信耗时/知识库注入）、
       「轮次→可信耗时(ncu kernel-only)」对数折线图（点色=瓶颈分类）、
       相对首轮加速比折线图、ncu 四指标(SM/DRAM/占用率/L2)分轮柱状图、逐轮明细表。
     - 可信计时口径统一用 ncu_time_sec（与论文口径一致），自报 Time 仅作对照列；NaN/损坏索引行安全降级。
     - test_dashboard.py：7 项离线单测全过（聚合排序、NaN 净化、空目录、损坏索引行容错、计时口径/知识库字段、
       run 级成功判定）；跨 run 对比只列成功 run，失败 run 单视图可查（❌失败 标记）。
     - 说明：原计划 Streamlit/Gradio 会引入较重依赖，与 rag.py「零依赖」路线冲突，故用 stdlib 等价实现；
       若后续要网页端直接发评测请求，再在 dashboard.py 加一个 `/api/run` 触发 async_eval（见阶段二 Auto-tuning）。
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
7. 引入 Auto-tuning (自动参数搜索) —— **本版重点展开，算法谱系 + 集成方案**

   · 目标：让大模型生成代码骨架，参数部分（block_size / grid_size / TILE / unroll 等）留空。
     系统自动遍历测试不同参数组合，找出最优解。算法与工程结合，榨干 RTX 5060 的最后一滴算力。

   7.0 参数空间（Search Space，所有算法的公共输入）
       把可调维度显式建模为「离散参数向量」，附硬约束与预算，作为每一级算法的统一接口：

       | 参数 | 类型 | 候选值（示例） | 硬约束 |
       |---|---|---|---|
       | BLOCK_M × BLOCK_N（shared tile） | 离散 | 16 / 32 / 64 / 128 | tile²×4B×2 ≤ shared 预算（sm_120 默认 48KB/block，opt-in ~99KB） |
       | BLOCK（每 block 线程数） | 离散 | 128 / 256 / 512 / 1024 | 32 倍数，≤1024 |
       | VECTOR（向量化加载宽度） | 离散 | 1 / 2 / 4（float4） | K 维对齐到向量宽度 |
       | UNROLL（内层 K 展开） | 离散 | 1 / 2 / 4 / 8 / full | 寄存器预算，spill 阈值内 |
       | REG_TILE（每线程输出分块） | 离散 | 1×1 / 2×2 / 4×4 | regs/thread ≤ 255 |
       | STAGES（双缓冲级数） | 离散 | 1 / 2 / 3 | 与 tile 共用 shared 预算 |
       | SPLIT_K（K 维切分） | 离散 | 1 / 2 / 4 / 8 | 输出 reduce 保持正确性 |

       · 说明：最坏 ~7 维小离散空间（几十到几百个合法组合），穷举/随机/爬山/BO 均可在分钟级跑完。
         与 micro_bench_unroll.py 现有「TILE × unroll × 内核族」受控扫描同构，直接复用其编译/计时/正确性基建。

   7.1 成本函数与噪声处理（统一口径）
       · fitness(params) = ncu 可信计时（kernel-only，秒）；ncu 不可用时回退 median 自报计时（与论文口径一致）。
       · 正确性 gate：与 float64 参考按尺寸校准容差比对，不合格记 +∞（不可行点，直接淘汰）。
       · 噪声：GPU 计时抖动 → 每点跑 TIMING_REPEATS 次取中位数；爬山/退火接受移动的阈值取 >2× 重复标准差，
         避免被噪声骗过（防「假提升」）。

   7.2 算法谱系（由简到繁，逐级对比，均可复现）

       ① Grid Search（穷举基线）
          · 全空间枚举直接求全局最优，作为所有算法的上界参照；无噪声假设下最稳。
          · 现状：micro_bench_unroll.py 已实现 TILE×unroll 受控扫描，升级为通用 grid 即可。

       ② Random Search（随机搜索）
          · 高维/预算受限时的强基线；给后续算法做「显著性下限」——若爬山/BO 打不过随机，说明结构没学对。

       ③ Hill Climbing（爬山）⭐ 首推落地
          · 流程：从种子点（LLM 骨架默认参数 或 best-so-far）出发，逐维 ±1 步长扰动生成邻域，
            取邻域最优点；仅接受严格更快（fitness 下降超过噪声阈值）。
          · 逃逸局部最优：多起点重启（multi-start）+ 禁忌表（tabu list，不回访已测点）。
          · 优点：实现 ~20 行、零导数、天然适配离散参数；缺点：易陷局部最优，必须配重启。
          · 落地建议：先做「坐标邻域 + 多起点 + tabu」的最小可跑版，作为论文第一个 search baseline。

       ④ Coordinate Descent（坐标下降）
          · 固定其余维，对单维做线搜索（枚举该维全部候选），逐维轮换至收敛。
          · 适合参数间弱耦合（UNROLL 与 STAGES 基本独立）；收敛快但可能停在坐标方向最优而非全局。

       ⑤ Simulated Annealing（模拟退火）
          · 以概率 exp(-Δ/T) 接受劣解，温度按几何降温 T ← T·α；前期逃离局部最优，后期收敛到全局。
          · 超参少（T0, α, 每温度步数），比 GA 更稳，适合做「随机性更强的爬山」对照。

       ⑥ Genetic Algorithm（遗传算法）
          · 整数编码染色体 = 参数向量；锦标赛选择 + 均匀交叉 + 逐维变异 + 精英保留。
          · 适合参数间有组合效应（tile × register tiling × double buffering 互相挤 shared/寄存器预算）。
          · 原 ROADMAP 已提及；与爬山/退火并列为一组「零代理模型」搜索法对比。

       ⑦ Bayesian Optimization（贝叶斯优化，TPE/GP）⭐ 推荐主力
          · 用 GP 或 TPE 代理模型拟合「参数→耗时」，采集函数 EI/UCB 决定下一个评估点。
          · 依据：每次 ncu 评估是秒级（贵），BO 用最少评估次数逼近最优——sample-efficiency 收益最大。
            Triton / TVM 的 autotuner 即用 TPE/进化 + 成本模型。
          · 落地：先手写轻量 TPE（或 optuna 单文件），不引重依赖；与 rag.py 零依赖路线一致。

       ⑧ Hyperband / Successive Halving（预算感知）
          · 先用少量 repeat 粗测大量配置，再对幸存者加测精测，按预算轮次淘汰。
          · 与 BO 正交：BO 决定「测哪个点」，Hyperband 决定「给多少预算测到多准」。

       ⑨ LLM-guided Warm Start（LLM 引导热启动）—— 与外环闭环焊接的关键
          · 用 LLM 骨架 + RAG 策略卡给每个参数一个先验（TILING_01 → TILE=16/32，
            REGISTER_TILING_01 → 2×2，OCCUPANCY_01 → BLOCK 先验），把搜索域从「全空间」缩到「高先验子空间」。
          · 再把 autotuner 找到的最优参数向量写回 prompt / 候选卡，形成外环：
            「LLM 提结构 → 搜索器填参 → 最优参数反哺 LLM → 自蒸馏进知识库」。
          · 这是本平台区别于普通 autotuner 的研究点：**用 LLM 先验替换随机/均匀先验**。

   7.3 集成与落盘
       · 新增 autotuner.py：输入 = 参数空间 + LLM 生成的参数化内核骨架（参数用模板占位）+ 搜索预算；
         输出 = 最优参数向量 + 收敛曲线。复用 micro_bench_unroll.py 的编译/计时/正确性/画像四个子步骤。
       · 每次评估落盘 results/{run_id}/autotune_trials.jsonl（params → fitness/正确性/ncu 指标），
         供论文画「算法收敛曲线」与「best-so-far 轨迹」。
       · 与 async_eval 的关系：当前「LLM 改代码」是**代码级搜索**，autotuner 是**参数级搜索**，
         两者串联为「外环改结构 + 内环调参数」；内环不占 API 配额，可跑得更密。

   7.4 消融实验设计（论文素材）
       · 三因素：算法（grid/random/hill/coord/anneal/GA/BO/hyperband）× 尺寸（256/512/1024/2048）× 预算（30/60/120 次评估）。
       · 指标：最优耗时、达到 95% 最优所需评估次数、跨尺寸可迁移性（256 上搜到的参数在 1024 上是否仍优）。
       · 预期结论（可证伪）：预算小 → BO/爬山胜；预算大 → grid 逼近；LLM 热启动在小预算下额外领先。

   7.5 附带议题（源自原 ROADMAP 尾部注释，纳入本项）
       · 不同 block size 对 kernel 耗时的影响 → 即 7.0 的 BLOCK 维，直接纳入搜索空间。
       · 引入 Unified Memory 简化内存分配 → 作为「工程简化」与「性能」的权衡项：先测 UM 的
         首次访问 page-fault 开销是否吃掉收益，数据说话再决定是否进骨架模板（避免盲目引入）。

---

🔴 阶段三：前沿探索与多智能体

此阶段对标科研级系统，甚至可以直接作为保研/直博的敲门砖。

8. Multi-Agent (多智能体) 协同系统
   · 完善：打造三个独立的 AI 角色。
     · Coder Agent：负责写代码。
     · Reviewer Agent：负责 Review 代码逻辑，检查有无死锁、越界风险。
     · Tester Agent：负责在沙箱中跑测试并提取数据。
   · 目的：让 AI 内部互相博弈、互相纠错，实现无人值守的代码进化。
   · 延伸（与第 7 项联动）：Tester Agent 跑出的参数级数据（autotune_trials.jsonl）作为
     Reviewer 判断「结构合理 + 参数未调优」的客观依据，避免 Reviewer 只靠直觉。
9. 算子融合 (Operator Fusion) 探索
   · 完善：不再局限于单个矩阵乘法。尝试让 AI 自动推导：能不能把 矩阵乘法 + ReLU激活函数 合并写在同一个 CUDA Kernel 里？
   · 目的：减少显存 I/O 搬运时间，这是当前大模型推理加速的核心瓶颈（如 FlashAttention 的核心思想）。
   · 延伸：融合后的内核把「中间张量落盘再读回」变成「寄存器内即算即用」，用第 7 项的
     参数搜索在融合内核上再跑一遍，量化「融合 + 调参」的叠加收益。
10. 探索形式化验证 (SMT Solver)
    · 完善：对于 AI 生成的代码，不仅靠测试用例跑分，还尝试用数学逻辑（如 Z3 求解器）证明其逻辑的绝对正确性。
    · 目的：解决 AI 在安全攸关（如自动驾驶）等底层代码生成中的信任问题。
    · 延伸：与第 7 项的正确性 gate 互补——测试用例做「大概率正确」过滤，SMT 做「边界条件证明」，
      先对「越界/tile 边界/SPLIT_K reduce 顺序」这三类高风险点做符号化，成本可控。
11. 编译器级调度搜索（MLIR / Triton 泛化）
    · 完善：把第 7 项的「参数级搜索」从手写 CUDA 推广到「调度空间（schedule space）」，
      让 LLM 生成 Triton/MLIR 的调度描述（而非裸 CUDA），搜索器在调度树/循环变换上做决策。
    · 目的：从「调几个 int 参数」升级到「调循环分块/重排/向量化的组合爆炸空间」，
      对标 Triton autotuner / TVM 的进化搜索；这是把 Demo 推向系统会议的关键一步。
12. 集群级演进（Distributed / NCCL）
    · 完善：突破单卡限制，向多卡 NCCL 通信、FlashAttention 等 Transformer 核心组件优化进军。
    · 目的：把「单算子自动调优」的成果复用到「通信-计算重叠」这一新搜索空间（环状 AllReduce、
      pipeline 并行切分），冲刺 MLSys / OSDI / ASPLOS 系统顶会。
