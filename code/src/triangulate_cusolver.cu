#include "triangulate.cuh"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <iostream>

// Helper macro for checking CUDA error codes
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Helper macro for checking cuSOLVER error codes
#define CUSOLVER_CHECK(call)                                                      \
    do {                                                                          \
        cusolverStatus_t status = call;                                           \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "CuSOLVER error at %s:%d: status %d\n",               \
                    __FILE__, __LINE__, status);                                  \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)


//preprocessing function to build A matrices
__global__ void build_A_matrix_kernel(
    const float *C1_d,      // 3x4
    const float *pts1_d,    // N x 2
    const float *C2_d,      // 3x4
    const float *pts2_d,    // N x 2
    float *A_batch_d,       // N * 16 (Output batched A matrices)
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // Pointer to the start of the current 4x4 matrix A_i
        float *A = A_batch_d + i * 16; 
        
        float u1 = pts1_d[i * 2 + 0];
        float v1 = pts1_d[i * 2 + 1];
        float u2 = pts2_d[i * 2 + 0];
        float v2 = pts2_d[i * 2 + 1];

        // C matrices are 3x4
        const float *C1_row0 = C1_d + 0 * 4;
        const float *C1_row1 = C1_d + 1 * 4;
        const float *C1_row2 = C1_d + 2 * 4;
        
        const float *C2_row0 = C2_d + 0 * 4;
        const float *C2_row1 = C2_d + 1 * 4;
        const float *C2_row2 = C2_d + 2 * 4;

        // A is stored in row-major order in the kernel.
        for (int j = 0; j < 4; j++) {
            A[0 * 4 + j] = u1 * C1_row2[j] - C1_row0[j];
            A[1 * 4 + j] = v1 * C1_row2[j] - C1_row1[j];
            A[2 * 4 + j] = u2 * C2_row2[j] - C2_row0[j];
            A[3 * 4 + j] = v2 * C2_row2[j] - C2_row1[j];
        }
    }
}


//Postprocessing function to get P points
__global__ void extract_P_kernel(
    const float *V_batch_d, // N * 16 (Batched V matrices from SVD)
    float *P_d,             // N * 3 (Output 3D points)
    int N                   // Batch size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        // Pointer to the current 4x4 V matrix
        const float *V = V_batch_d + i * 16;
        
        // cuSOLVER returns V in column-major order
        // The null vector is the last column of V.
        // P_homo[0] = V[3] 
        // P_homo[1] = V[7]
        // P_homo[2] = V[11]
        // P_homo[3] = V[15]
        
        float P_homo_W = V[15];
        
        if (fabs(P_homo_W) > 1e-6) {
            float inv_W = 1.0f / P_homo_W;
            
            // X = X/W, Y = Y/W, Z = Z/W
            P_d[i * 3 + 0] = V[3] * inv_W;   // X / W
            P_d[i * 3 + 1] = V[7] * inv_W;   // Y / W
            P_d[i * 3 + 2] = V[11] * inv_W;  // Z / W
        } else {
            // Cannot normalize (point at infinity or numerical error)
            P_d[i * 3 + 0] = 0.0f;
            P_d[i * 3 + 1] = 0.0f;
            P_d[i * 3 + 2] = 0.0f;
        }
    }
}


//Host Side triangulate function
float cuda_triangulate(
    const std::vector<float>& C1,
    const std::vector<float>& pts1,
    const std::vector<float>& C2,
    const std::vector<float>& pts2,
    std::vector<float>& P_out
) {
    //variable setup for Preprocessing, SVD, and Postprocessing
    int N = pts1.size() / 2; 
    const int M = 4;    // Height of A
    const int K = 4;    // Width of A
    const int rank = 4; // 4 Singular vectors
    const int LDA = M;  // Leading dimension of A = M = 4
    const int LDU = M;  // Leading dimension of U = M = 4
    const int LDV = K;  // Leading dimension of V = Non-Leading of A = K = 4
    const long long int strideA = M * K; // Stride is 16 elements since A is 4x4
    
    // Total device memory needed for N batches of 4x4 matrices
    size_t batch_size_bytes = (size_t)N * strideA * sizeof(float); 
    size_t pts_size_bytes = pts1.size() * sizeof(float);
    size_t C_size_bytes = C1.size() * sizeof(float);
    size_t P_out_size_bytes = (size_t)N * 3 * sizeof(float);
    
    // Device pointers
    float *C1_d, *pts1_d, *C2_d, *pts2_d;
    float *A_batch_d = nullptr; // Input matrices A
    float *S_batch_d = nullptr; // Output S 
    float *V_batch_d = nullptr; // Output V matrices
    float *d_work = nullptr;    // CuSOLVER work buffer
    int *info_d = nullptr;      // CuSOLVER status info
    float *U_batch_d = nullptr; // U is required to be allocated for the function call
    
    // Host pointer for the Frobenius norm residual 
    double h_R_nrmF = 0.0;

    // CuSOLVER handle
    cusolverDnHandle_t cusolverH = nullptr;
    int lwork = 0; // The size of the workspace, queried first
    
    //allocations
    CUDA_CHECK(cudaSetDevice(0));
    
    CUDA_CHECK(cudaMalloc((void**)&C1_d, C_size_bytes));
    CUDA_CHECK(cudaMemcpy(C1_d, C1.data(), C_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&C2_d, C_size_bytes));
    CUDA_CHECK(cudaMemcpy(C2_d, C2.data(), C_size_bytes, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc((void**)&pts1_d, pts_size_bytes));
    CUDA_CHECK(cudaMemcpy(pts1_d, pts1.data(), pts_size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&pts2_d, pts_size_bytes));
    CUDA_CHECK(cudaMemcpy(pts2_d, pts2.data(), pts_size_bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&A_batch_d, batch_size_bytes));
    CUDA_CHECK(cudaMalloc((void**)&V_batch_d, batch_size_bytes)); 
    CUDA_CHECK(cudaMalloc((void**)&S_batch_d, (size_t)N * K * sizeof(float))); 
    CUDA_CHECK(cudaMalloc((void**)&info_d, (size_t)N * sizeof(int)));      
    CUDA_CHECK(cudaMalloc((void**)&U_batch_d, batch_size_bytes)); 

    //build A matrices
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    build_A_matrix_kernel<<<numBlocks, threadsPerBlock>>>(
        C1_d, pts1_d, C2_d, pts2_d, A_batch_d, N
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //SVD
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    
    //helper function from cuSOLVER to compute the needed buffer size, saves in lwork
    //arguments are identical to cusolverDnSgesvdaStridedBatched
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
        cusolverH, CUSOLVER_EIG_MODE_VECTOR, rank, 
        M, K, A_batch_d, LDA, strideA, S_batch_d, 
        strideA, U_batch_d, LDU, strideA, V_batch_d, LDV, strideA, &lwork, N
    ));

    // Allocate the necessary workspace on the device (size is lwork)
    CUDA_CHECK(cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float)));

    // Perform the StridedBatched SVD: A = U * S * V^T
    CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
        cusolverH, 
        CUSOLVER_EIG_MODE_VECTOR,   // jobz: Compute all vectors (U and V)
        rank,                       // rank: Number of singular values/vectors to compute (4)
        M, K,                       // M=4, N=4 (Rows/Columns)
        A_batch_d, LDA, strideA,    // A data pointer, LDA, stride
        S_batch_d, strideA,         // S data pointer, stride
        U_batch_d, LDU, strideA,    // U data pointer, LDU, stride
        V_batch_d, LDV, strideA,    // V data pointer, LDV, stride
        d_work, lwork,              // Workspace and size, computed already
        info_d, 
        &h_R_nrmF,                  // Host residual norm (double pointer)
        N                           // Batch size N
    ));
    
    CUDA_CHECK(cudaDeviceSynchronize());

    //free up space cus line 222 has a memory issue
    cudaFree(C1_d);
    cudaFree(pts1_d);
    cudaFree(C2_d);
    cudaFree(pts2_d);
    cudaFree(A_batch_d);
    cudaFree(U_batch_d);
    cudaFree(S_batch_d);
    cudaFree(d_work);
    cudaFree(info_d);

    //LINE 222 CURRENTLY NOT WORKING CUS OF A OOM ISSUE
    float *P_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&P_d, P_out_size_bytes));
    
    extract_P_kernel<<<numBlocks, threadsPerBlock>>>(
        V_batch_d, P_d, N 
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //copy final points to host
    P_out.resize(N * 3);
    CUDA_CHECK(cudaMemcpy(P_out.data(), P_d, P_out_size_bytes, cudaMemcpyDeviceToHost));
    
    //destroy the solver
    if (cusolverH) CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    
    // Cleanup Remaining Device Memory
    
    cudaFree(V_batch_d);
    cudaFree(P_d);

    return 0.0f; 
}