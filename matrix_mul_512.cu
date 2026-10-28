#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

#define N 512
#define TILE 16

__global__ void matmul(const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    for (int m = 0; m < N / TILE; ++m) {
        As[ty][tx] = A[row * N + m * TILE + tx];
        Bs[ty][tx] = B[(m * TILE + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[row * N + col] = sum;
}

int main() {
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i)
        matmul<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    double per_run = std::chrono::duration<double>(end - start).count() / 100.0;
    std::cout << "Time: " << per_run << " seconds" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}