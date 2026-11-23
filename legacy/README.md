# legacy

早期脚本，**已弃用，仅保留作历史参考**，不建议在新流程中使用：

| 文件 | 说明 |
|---|---|
| `auto_test.sh` | 大创初期的 C 基准 + Docker + torch 一键评测；需要根目录存在 `matrix_mul.cu` |
| `test_api.py` | 第一版“生成 → 编译 → 校验 → 反馈”脚本，功能已被 `async_eval.py` 覆盖 |
| `test_concurrency.py` | 并发信号量极限测试，已被 `test_loop_injection.py` / `test_distill_loop.py` 覆盖 |

当前主流程：

```bash
python preflight.py
python run_experiment.py --exp exp1 --kb 0,1 --repeats 3
```
