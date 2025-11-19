#include "cuda_ops.cuh"

// 1. CUDA Kernel (Device Code)
// Multiplies two arrays element-wise and stores the result in C
__global__ void multiply_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

// 2. C++ Host Wrapper Function
// This function will be called from Python
void launch_multiply_kernel(const float* A_host, const float* B_host, float* C_host, int N) {
    float *A_dev, *B_dev, *C_dev;
    size_t size = N * sizeof(float);

    // 1. Allocate device memory
    cudaMalloc((void**)&A_dev, size);
    cudaMalloc((void**)&B_dev, size);
    cudaMalloc((void**)&C_dev, size);

    // 2. Copy data from Host (CPU) to Device (GPU)
    cudaMemcpy(A_dev, A_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, size, cudaMemcpyHostToDevice);

    // 3. Configure and Launch the Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev, N);

    // 4. Copy results from Device (GPU) back to Host (CPU)
    cudaMemcpy(C_host, C_dev, size, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}