/*
triangulate function using the One-Sided Jacobi Method
Idea proposed by google gemini.
This idea was used instead of the cuSOLVER implementation as the cuSOLVER SVD 
algorithms are very strong and robust, but are meant for very very large 
matrices, using many global variables that slow down performance compared to a 
CPU implementation on small inputs.

citation: 
Hestenes, M. R. (1958). "Inversion of Matrices by Biorthogonalization and Related Results." 
Journal of the Society for Industrial and Applied Mathematics, 6(1), 51–90.
*/ 

#include "triangulate.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cfloat> // For FLT_MAX

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// --- device helper functions ---

//build A matrix, values stored in local registers
__device__ void build_A_local(
    const float *C1, const float *C2,
    float u1, float v1, float u2, float v2,
    float A[16]
) {
    // Row-major
    for (int j = 0; j < 4; j++) {
        A[0*4+j] = u1*C1[2*4+j] - C1[0*4+j]; // Row 0: u1*P3 - P1
        A[1*4+j] = v1*C1[2*4+j] - C1[1*4+j]; // Row 1: v1*P3 - P2
        A[2*4+j] = u2*C2[2*4+j] - C2[0*4+j]; // Row 2: u2*P3' - P1'
        A[3*4+j] = v2*C2[2*4+j] - C2[1*4+j]; // Row 3: v2*P3' - P2'
    }
}

// Compute SVD using One-Sided Jacobi on registers
// A becomes (approx U*Sigma)
// V becomes the right singular vectors
__device__ void svd_4x4_jacobi_local(float A[16], float V[16]) {
    // Initialize V
    V[0]=1.0f; V[1]=0.0f; V[2]=0.0f; V[3]=0.0f;
    V[4]=0.0f; V[5]=1.0f; V[6]=0.0f; V[7]=0.0f;
    V[8]=0.0f; V[9]=0.0f; V[10]=1.0f; V[11]=0.0f;
    V[12]=0.0f; V[13]=0.0f; V[14]=0.0f; V[15]=1.0f;

    //4 iters works decently well for a 4x4 matrix
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        //iterate over all pairs p and q
        for (int p = 0; p < 3; p++) {
            for (int q = p + 1; q < 4; q++) {
                float a = 0.0f, b = 0.0f, d = 0.0f;

                //dot products
                for (int i = 0; i < 4; i++) {
                    float val_p = A[p + 4*i]; 
                    float val_q = A[q + 4*i]; 
                    a += val_p * val_p;
                    b += val_q * val_q;
                    d += val_p * val_q;
                }

                if (fabsf(d) < 1e-9f) continue; //skip if already orthogonal

                //givens rotation
                float tau = (b - a) / (2.0f * d);
                float sign = (tau >= 0.0f) ? 1.0f : -1.0f;
                float t = sign / (fabsf(tau) + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = c * t;

                //apply rotation
                for (int i = 0; i < 4; i++) {
                    
                    //update A
                    int idx_p = p + 4*i;
                    int idx_q = q + 4*i;
                    float Ap = A[idx_p];
                    float Aq = A[idx_q];
                    A[idx_p] = c * Ap - s * Aq;
                    A[idx_q] = s * Ap + c * Aq;

                    //update V
                    float Vp = V[idx_p];
                    float Vq = V[idx_q];
                    V[idx_p] = c * Vp - s * Vq;
                    V[idx_q] = s * Vp + c * Vq;
                }
            }
        }
    }
}

// Extract P by finding the singular vector corresponding to the smallest singular value
__device__ void extract_P_local(float A[16], float V[16], float *P_row_ptr) {
    int min_col = 0;
    float min_norm_sq = FLT_MAX;

    //find which col has the smallest norm
    //corresponds to the smallest singular value
    for (int j = 0; j < 4; j++) {
        float col_norm_sq = 0.0f;
        for (int i = 0; i < 4; i++) {
            float val = A[j + 4*i];
            col_norm_sq += val * val;
        }
        
        if (col_norm_sq < min_norm_sq) {
            min_norm_sq = col_norm_sq;
            min_col = j;
        }
    }

    //get corresponding column (V is row major)
    float X = V[min_col + 4*0];
    float Y = V[min_col + 4*1];
    float Z = V[min_col + 4*2];
    float W = V[min_col + 4*3];

    //perspective division
    if (fabsf(W) > 1e-8f) {
        float inv_W = 1.0f / W;
        P_row_ptr[0] = X * inv_W;
        P_row_ptr[1] = Y * inv_W;
        P_row_ptr[2] = Z * inv_W;
    } else {
        P_row_ptr[0] = 0.0f;
        P_row_ptr[1] = 0.0f;
        P_row_ptr[2] = 0.0f;
    }
}

//--- combined kernel function ---
__global__ void triangulate_fused_kernel(
    const float *C1_d, const float *pts1_d,
    const float *C2_d, const float *pts2_d,
    float *P_d, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    //data stored in thread local registers which allow for high performance over cusolver
    float A[16];
    float V[16];

    //Load inputs
    float u1 = pts1_d[i*2];
    float v1 = pts1_d[i*2+1];
    float u2 = pts2_d[i*2];
    float v2 = pts2_d[i*2+1];

    //build A
    build_A_local(C1_d, C2_d, u1, v1, u2, v2, A);

    //compute SVD using jacobi methods
    svd_4x4_jacobi_local(A, V);

    //get P
    extract_P_local(A, V, P_d + i*3);
}

// --- Host function ---
std::vector<float> cuda_triangulate(
    const std::vector<float>& C1, const std::vector<float>& pts1,
    const std::vector<float>& C2, const std::vector<float>& pts2
) {
    int N = pts1.size() / 2;
    if (N == 0) return std::vector<float>();

    float *C1_d, *C2_d, *pts1_d, *pts2_d, *P_d;

    // Allocate Buffers
    CUDA_CHECK(cudaMalloc(&C1_d, 48)); 
    CUDA_CHECK(cudaMalloc(&C2_d, 48));
    CUDA_CHECK(cudaMalloc(&pts1_d, pts1.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pts2_d, pts2.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P_d, N * 3 * sizeof(float)));
    
    // Copy Data
    CUDA_CHECK(cudaMemcpy(C1_d, C1.data(), 48, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C2_d, C2.data(), 48, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pts1_d, pts1.data(), pts1.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pts2_d, pts2.data(), pts2.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch Kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // One big parallel kernel launch
    triangulate_fused_kernel<<<blocks, threads>>>(C1_d, pts1_d, C2_d, pts2_d, P_d, N);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy Results to host
    std::vector<float> result(N * 3);
    CUDA_CHECK(cudaMemcpy(result.data(), P_d, N * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free Memory
    cudaFree(C1_d); cudaFree(C2_d); cudaFree(pts1_d); cudaFree(pts2_d);
    cudaFree(P_d);
    
    return result;
}
