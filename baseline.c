#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

int main() {
    // 1. 动态分配内存（1024x1024 比较大，必须用 malloc 放在堆区，否则栈会溢出）
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C = (int *)malloc(N * N * sizeof(int));

    if (A == NULL || B == NULL || C == NULL) {
        printf("内存分配失败！\n");
        return 1;
    }

    // 2. 初始化矩阵（赋值一些简单的随机数）
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i + j) % 10;
            B[i * N + j] = (i - j) % 10;
            C[i * N + j] = 0;
        }
    }

    // 3. 开始计时
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 4. 纯 C 的三层循环矩阵乘法（这就是 Baseline 的核心）
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }

    // 5. 结束计时
    clock_gettime(CLOCK_MONOTONIC, &end);

    // 6. 计算耗时（毫秒）
    double time_spent = (end.tv_sec - start.tv_sec) * 1000.0 + 
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;

    // 7. 打印结果和耗时（打印 C[0][0] 用于人工校对，防止 AI 算错了）
    printf("纯 C 语言计算 %dx%d 矩阵乘法完成。\n", N, N);
    printf("校验值 C[0][0] = %d\n", C[0]);
    printf("纯 C 耗时: %.2f ms\n", time_spent);

    // 8. 释放内存
    free(A);
    free(B);
    free(C);

    return 0;
}
