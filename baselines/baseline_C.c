#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

int main() {
    // float32，与 CUDA 核函数、torch.matmul 对齐（不再用 int）
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));

    if (A == NULL || B == NULL || C == NULL) {
        printf("内存分配失败！\n");
        return 1;
    }

    // 统一确定性初始化（与 baseline_torch.py 及 AI 生成核函数保持一致）
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = ((i + j) % 10) * 0.1f;
            B[i * N + j] = ((i * 3 + j) % 10) * 0.1f;
            C[i * N + j] = 0.0f;
        }
    }

    // 开始计时（CPU 墙钟）
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 纯 C 三层循环，float 累加（Baseline 核心）
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // 结束计时
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_nsec - start.tv_nsec) / 1000000.0;

    // 将结果矩阵以二进制 dump（float32 行优先，N*N 个元素），供 Python 用 numpy.allclose 校验
    FILE *fp = fopen("c_output.bin", "wb");
    if (fp == NULL) {
        printf("错误：无法写入 c_output.bin\n");
    } else {
        fwrite(C, sizeof(float), (size_t)N * N, fp);
        fclose(fp);
    }

    // 打印校验值 C[0][0]（与 torch/CUDA 同口径可交叉验证）与耗时
    printf("纯 C 计算 %dx%d 矩阵乘法完成（float32）。\n", N, N);
    printf("C[0][0] = %.6f\n", C[0]);
    printf("纯 C 耗时: %.2f ms\n", time_ms);
    printf("Time: %.6f seconds\n", time_ms / 1000.0);

    free(A);
    free(B);
    free(C);

    return 0;
}
