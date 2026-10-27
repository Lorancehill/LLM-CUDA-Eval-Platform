#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdio>

const int N = 1024;
const int TILE = 16;

__global__ void matmul(const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[ty][tx] = A[row * N + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    C[row * N + col] = sum;
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C: %s\n", cudaGetErrorString(err)); return 1; }

    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "memcpy A: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "memcpy B: %s\n", cudaGetErrorString(err)); return 1; }

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    auto start = std::chrono::high_resolution_clock::now();

    matmul<<<grid, block>>>(d_A, d_B, d_C);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "kernel launch: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel exec: %s\n", cudaGetErrorString(err)); return 1; }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;

    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "memcpy C: %s\n", cudaGetErrorString(err)); return 1; }

    bool ok = true;
    float expected = (float)N * 2.0f;
    for (int i = 0; i < N * N; ++i) {
        if (h_C[i] != expected) { ok = false; break; }
    }
    std::cout << "Verify: " << (ok ? "PASS" : "FAIL") << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}