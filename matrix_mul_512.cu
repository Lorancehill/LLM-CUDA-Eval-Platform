#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <random>

#define N 512
#define TILE 16

__global__ void matMul(const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    for (int t = 0; t < N / TILE; ++t) {
        As[ty][tx] = A[row * N + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    C[row * N + col] = sum;
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *hA = new float[N * N];
    float *hB = new float[N * N];
    float *hC = new float[N * N];

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        hA[i] = dist(gen);
        hB[i] = dist(gen);
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err;
    err = cudaMalloc((void**)&dA, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dA: " << cudaGetErrorString(err) << std::endl; return 1; }
    err = cudaMalloc((void**)&dB, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dB: " << cudaGetErrorString(err) << std::endl; return 1; }
    err = cudaMalloc((void**)&dC, bytes);
    if (err != cudaSuccess) { std::cerr << "cudaMalloc dC: " << cudaGetErrorString(err) << std::endl; return 1; }

    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    auto start = std::chrono::high_resolution_clock::now();
    matMul<<<grid, block>>>(dA, dB, dC);
    err = cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time: " << elapsed.count() << " seconds" << std::endl;

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;
}