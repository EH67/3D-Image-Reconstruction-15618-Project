/**
* DEPRECATED - triangulate implementation using cuSOLVER.
*
* This was one of our approaches for solving the SVDs. However, it actually was slower
* than just doing a serial SVD implementation and assign it to 1 GPU thread since the
* matrices we are working with were really small (3*3, 4*4, 8*9, etc), and it is not worth
* the global memory accesses and synchronization that cuSOLVER does. More details are given
* in the final report.
*/

#include "triangulate.cuh"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <cstdio>
#include <algorithm> // For std::min

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Updated to print the specific error code
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSOLVER Error at line %d: Status %d\n", __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


//preprocessing function ot build A's for svd call
__global__ void build_A_matrix_kernel(
    const float *C1_d, const float *pts1_d,
    const float *C2_d, const float *pts2_d,
    float *A_batch_d, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    // Pointer to the start of the current 4x4 matrix A_i
    float *A = A_batch_d + i * 16;
    
    float u1 = pts1_d[i*2];
    float v1 = pts1_d[i*2+1];

    float u2 = pts2_d[i*2];
    float v2 = pts2_d[i*2+1];
    
    // Row-major construction
    for (int j = 0; j < 4; j++) {
        A[j*4+0] = u1*C1_d[2*4+j] - C1_d[0*4+j];
        A[j*4+1] = v1*C1_d[2*4+j] - C1_d[1*4+j];
        A[j*4+2] = u2*C2_d[2*4+j] - C2_d[0*4+j];
        A[j*4+3] = v2*C2_d[2*4+j] - C2_d[1*4+j];
    }
}

//postprocessing function to compute final 3d points
__global__ void extract_P_kernel(
    const float *V_batch_d, float *P_d, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const float *V = V_batch_d + i * 16;
    float W = V[15]; // Last element (index 15)
    
    if (fabs(W) > 1e-8f) {
        float inv_W = 1.0f / W;
        P_d[i*3+0] = V[12] * inv_W;
        P_d[i*3+1] = V[13] * inv_W;
        P_d[i*3+2] = V[14] * inv_W;
    } else {
        P_d[i*3+0] = 0.0f;
        P_d[i*3+1] = 0.0f;
        P_d[i*3+2] = 0.0f;
    }
}

std::vector<float> cuda_triangulate(
    const std::vector<float>& C1, const std::vector<float>& pts1,
    const std::vector<float>& C2, const std::vector<float>& pts2
) {
    int N = pts1.size() / 2;
    if (N == 0) return std::vector<float>();

    float *C1_d, *C2_d, *pts1_d, *pts2_d, *A_d, *V_d, *U_d, *S_d, *P_d, *work;
    int *info;

    // Allocate Global Memory
    // We allocate all data buffers at once to avoid repeated malloc overhead.
    // If there is too much data, this may need to be done in batches but
    // this was tested with 1,000,000 datapoints and was fine
    CUDA_CHECK(cudaMalloc(&C1_d, 48));
    CUDA_CHECK(cudaMalloc(&C2_d, 48));
    CUDA_CHECK(cudaMalloc(&pts1_d, pts1.size()*4));
    CUDA_CHECK(cudaMalloc(&pts2_d, pts2.size()*4));
    CUDA_CHECK(cudaMalloc(&A_d, N*64));     // 16 floats * N
    CUDA_CHECK(cudaMalloc(&V_d, N*64));     // 16 floats * N
    CUDA_CHECK(cudaMalloc(&U_d, N*64));     // 16 floats * N
    CUDA_CHECK(cudaMalloc(&S_d, N*16));     // 4 floats * N
    CUDA_CHECK(cudaMalloc(&info, N*4));     // 1 int * N
    CUDA_CHECK(cudaMalloc(&P_d, N*12));     // 3 floats * N (output)
    
    // Copy Inputs to device
    CUDA_CHECK(cudaMemcpy(C1_d, C1.data(), 48, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C2_d, C2.data(), 48, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pts1_d, pts1.data(), pts1.size()*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pts2_d, pts2.data(), pts2.size()*4, cudaMemcpyHostToDevice));
    
    //build A matrix kernal
    int blocks = (N + 255) / 256;
    build_A_matrix_kernel<<<blocks, 256>>>(C1_d, pts1_d, C2_d, pts2_d, A_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //setup cuSOLVER
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    
    //Batching to prevent segfaulting and other wierd behavior
    const int BATCH_SIZE = 200000; 
    int lwork = 0;
    
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        handle, 
        CUSOLVER_EIG_MODE_VECTOR, 
        4, 4, 4,               // rank, m, n
        A_d, 4, 16LL,          // stride parameters (dummy ptrs ok for size check)
        S_d, 4LL, 
        U_d, 4, 16LL, 
        V_d, 4, 16LL, 
        &lwork, 
        BATCH_SIZE             // Query size for the MAX usable batch size
    ));
    
    CUDA_CHECK(cudaMalloc(&work, lwork * sizeof(float)));

    //execute in batches to prevent stack smashing and OOM issues
    for (int offset = 0; offset < N; offset += BATCH_SIZE) {
        int current_batch = std::min(BATCH_SIZE, N - offset);
        
        //pointer arithmetic
        float *A_curr = A_d + (offset * 16);
        float *S_curr = S_d + (offset * 4);
        float *U_curr = U_d + (offset * 16);
        float *V_curr = V_d + (offset * 16);
        int   *info_curr = info + offset;

        CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
            handle, 
            CUSOLVER_EIG_MODE_VECTOR, 
            4, 4, 4, 
            A_curr, 4, 16LL, 
            S_curr, 4LL, 
            U_curr, 4, 16LL, 
            V_curr, 4, 16LL, 
            work, lwork, 
            info_curr, 
            nullptr, // rn_out is null
            current_batch
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //get P
    extract_P_kernel<<<blocks, 256>>>(V_d, P_d, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    //copy result
    float *host_buf;
    CUDA_CHECK(cudaMallocHost(&host_buf, N*12));
    CUDA_CHECK(cudaMemcpy(host_buf, P_d, N*12, cudaMemcpyDeviceToHost));
    
    std::vector<float> result(host_buf, host_buf + N*3);
    
    //free everything
    cudaFreeHost(host_buf);
    cusolverDnDestroy(handle);
    cudaFree(C1_d); cudaFree(C2_d); cudaFree(pts1_d); cudaFree(pts2_d);
    cudaFree(A_d); cudaFree(V_d); cudaFree(U_d); cudaFree(S_d);
    cudaFree(info); cudaFree(work); cudaFree(P_d);
    
    return result;
}
