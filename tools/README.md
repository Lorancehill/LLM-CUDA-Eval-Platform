# tools

辅助工具（不属于评测主流程）：

| 文件 | 说明 |
|---|---|
| `micro_bench_unroll.py` | 循环展开微基准：固定 matmul kernel × unroll 因子，容器编译 + 宿主 ncu，结果写 `experiments/micro_unroll/` |

```bash
python tools/micro_bench_unroll.py
python tools/micro_bench_unroll.py --only tiled2_32_u4
```
