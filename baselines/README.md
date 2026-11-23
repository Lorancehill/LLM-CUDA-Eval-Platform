# baselines

手写基准代码，用于和 LLM 生成的 CUDA kernel 做对照：

| 文件 | 说明 |
|---|---|
| `baseline_C.c` | 纯 C 三重循环矩阵乘法，`gcc -O3` 编译；与 prompt 相同的确定性初始化 |
| `baseline_torch.py` | PyTorch `torch.matmul`（CUDA）基准，需要 GPU 与 torch |

注意：`baseline_torch.py` 不在 `requirements.txt` 的默认依赖里，需要单独安装 torch。
