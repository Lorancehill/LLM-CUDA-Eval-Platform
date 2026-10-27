#include <cstdio>
#include <chrono>

#define N 256
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
    size_t bytes = N * N * sizeof(float);
    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }
    float *dA = 0, *dB = 0, *dC = 0;
    cudaMalloc((void**)&dA, bytes);
    cudaMalloc((void**)&dB, bytes);
    cudaMalloc((void**)&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    auto start = std::chrono::high_resolution_clock::now();
    matMul<<<grid, block>>>(dA, dB, dC);
    cudaError_t err = cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
    std::chrono::duration<double> elapsed = end - start;
    printf("Time: %f seconds\n", elapsed.count());

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;
}