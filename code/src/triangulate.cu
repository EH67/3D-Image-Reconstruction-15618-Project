#include "triangulate.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <float.h>


//determinant of a 3x3,
//pulled from https://www.geeksforgeeks.org/maths/determinant-of-a-matrix-formula/
__device__ float determinant_3x3(const float *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7]) - 
           M[1] * (M[3] * M[8] - M[5] * M[6]) + 
           M[2] * (M[3] * M[7] - M[4] * M[6]);
}

//find the inverse of the matrix using adjugates
//requires the determinant to be nonzero
__device__ float inverse_3x3(const float *M, float *Minv) {

    Minv[0] = M[4] * M[8] - M[5] * M[7];
    Minv[1] = M[2] * M[7] - M[1] * M[8];
    Minv[2] = M[1] * M[5] - M[2] * M[4];
    Minv[3] = M[5] * M[6] - M[3] * M[8];
    Minv[4] = M[0] * M[8] - M[2] * M[6];
    Minv[5] = M[2] * M[3] - M[0] * M[5];
    Minv[6] = M[3] * M[7] - M[4] * M[6];
    Minv[7] = M[1] * M[6] - M[0] * M[7];
    Minv[8] = M[0] * M[4] - M[1] * M[3];

    float det = M[0] * Minv[0] + M[1] * Minv[3] + M[2] * Minv[6];

    if (fabs(det) > FLT_EPSILON) {
        float invDet = 1.0f / det;
        for (int i = 0; i < 9; i++) {
            Minv[i] *= invDet;
        }
        return det;
    } 
    
    // Singular matrix
    return 0.0f;
}

__device__ void mat_vec_mult_3x3(const float *M, const float *v_in, float *v_out) {
    for (int i = 0; i < 3; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 3; j++) {
            sum += M[i * 3 + j] * v_in[j];
        }
        v_out[i] = sum;
    }
}

//DLT function to find the null space ofa. 4x4 matrix
__device__ bool solve_dlt_case(const float *A_sub, float *P_out) {
    
    // A' is the top-left 3x3 of A_sub. b is the 4th column of A_sub.
    float A_prime[9]; // 3x3 matrix A'
    float b[3];       // 3x1 vector b
    
    //populate A' and b
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A_prime[i * 3 + j] = A_sub[i * 4 + j];
        }
        b[i] = A_sub[i * 4 + 3]; // The 4th column (corresponding to W=1)
    }
    
    //invrese A'
    float A_prime_inv[9];
    float det = inverse_3x3(A_prime, A_prime_inv);

    //if singular
    if (fabs(det) < 1e-6f) {
        return false;
    }
    
    //X = -A_prime_inv * b
    float neg_b[3];
    for(int i=0; i<3; i++){
        neg_b[i] = -b[i];
    }
    
    float X[3]; 
    mat_vec_mult_3x3(A_prime_inv, neg_b, X);

    //fill P
    P_out[0] = X[0];
    P_out[1] = X[1];
    P_out[2] = X[2];
    
    return true;
}


//solving for potential 3d points using dlt
//initial plan uses svd but it is very involved and complex.
__device__ bool solve_robust_dlt_triangulation(const float *A, float *P_out) {
    
    float P_results[4][3]; // Storage for the 4 potential 3D points
    int success_count = 0;
    float sum_P[3] = {0.0f, 0.0f, 0.0f};

    int row_indices[4][3] = {
        {1, 2, 3}, // Drop Row 0
        {0, 2, 3}, // Drop Row 1
        {0, 1, 3}, // Drop Row 2
        {0, 1, 2}  // Drop Row 3
    };

    // Iterate through the 4 sub-problems
    for (int k = 0; k < 4; k++) {
        float A_sub[12]; // 3x4 sub-matrix
        
        for (int r = 0; r < 3; r++) {
            int original_row = row_indices[k][r];
            for (int c = 0; c < 4; c++) {
                A_sub[r * 4 + c] = A[original_row * 4 + c];
            }
        }
        
        if (solve_dlt_case(A_sub, P_results[k])) {
            //add to running sum
            sum_P[0] += P_results[k][0];
            sum_P[1] += P_results[k][1];
            sum_P[2] += P_results[k][2];
            success_count++;
        }
    }

    if (success_count > 0) {
        // Average the solutions
        float inv_count = 1.0f / (float)success_count;
        P_out[0] = sum_P[0] * inv_count;
        P_out[1] = sum_P[1] * inv_count;
        P_out[2] = sum_P[2] * inv_count;
        return true;
    }
    
    return false; // All 4 cases failed
}

__global__ void triangulation_kernel_gpu(const float *C1_d, const float *pts1_d,// 3x4, N x 2
                                         const float *C2_d, const float *pts2_d,// 3x4, N x 2
                                         float *P_d,                            // N x 3
                                         int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float A[16]; //4x4 observation matrix
        
        float u1 = pts1_d[i * 2 + 0];
        float v1 = pts1_d[i * 2 + 1];
        float u2 = pts2_d[i * 2 + 0];
        float v2 = pts2_d[i * 2 + 1];

        //3x4 c matrices
        const float *C1_row0 = C1_d + 0 * 4;
        const float *C1_row1 = C1_d + 1 * 4;
        const float *C1_row2 = C1_d + 2 * 4;
        
        const float *C2_row0 = C2_d + 0 * 4;
        const float *C2_row1 = C2_d + 1 * 4;
        const float *C2_row2 = C2_d + 2 * 4;

        //A[0]
        for (int j = 0; j < 4; j++) {
            A[0 * 4 + j] = u1 * C1_row2[j] - C1_row0[j];
        }
        // A[1]
        for (int j = 0; j < 4; j++) {
            A[1 * 4 + j] = v1 * C1_row2[j] - C1_row1[j];
        }
        // A[2]
        for (int j = 0; j < 4; j++) {
            A[2 * 4 + j] = u2 * C2_row2[j] - C2_row0[j];
        }
        // A[3]
        for (int j = 0; j < 4; j++) {
            A[3 * 4 + j] = v2 * C2_row2[j] - C2_row1[j];
        }
        
        //DLT for finding points
        float P_non_homo[3];
        bool success = solve_robust_dlt_triangulation(A, P_non_homo);

        if (success) {
            P_d[i * 3 + 0] = P_non_homo[0];
            P_d[i * 3 + 1] = P_non_homo[1];
            P_d[i * 3 + 2] = P_non_homo[2];
        } else {
            //failed
            P_d[i * 3 + 0] = 0.0f;
            P_d[i * 3 + 1] = 0.0f;
            P_d[i * 3 + 2] = 0.0f;
        }
    }
}


//cpu side function to prepare and launch triangulate kernel
float cuda_triangulate(const std::vector<float>& C1,const std::vector<float>& pts1,
                       const std::vector<float>& C2, const std::vector<float>& pts2,
                       std::vector<float>& P_out) {
    int N = pts1.size() / 2; 
    size_t pts_size = pts1.size() * sizeof(float);
    size_t C_size = C1.size() * sizeof(float);
    size_t P_out_size_bytes = N * 3 * sizeof(float);
    
    float *C1_d, *pts1_d, *C2_d, *pts2_d;
    float *P_d;

    cudaSetDevice(0);
    
    //camera1
    cudaMalloc((void**)&C1_d, C_size);
    cudaMemcpy(C1_d, C1.data(), C_size, cudaMemcpyHostToDevice);

    //camera2
    cudaMalloc((void**)&C2_d, C_size);
    cudaMemcpy(C2_d, C2.data(), C_size, cudaMemcpyHostToDevice);
    
    //points1
    cudaMalloc((void**)&pts1_d, pts_size);
    cudaMemcpy(pts1_d, pts1.data(), pts_size, cudaMemcpyHostToDevice);

    //points2
    cudaMalloc((void**)&pts2_d, pts_size);
    cudaMemcpy(pts2_d, pts2.data(), pts_size, cudaMemcpyHostToDevice);
    
    //output 3d points
    cudaMalloc((void**)&P_d, P_out_size_bytes);
    cudaMemset(P_d, 0, P_out_size_bytes);

    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    triangulation_kernel_gpu<<<numBlocks, threadsPerBlock>>>(C1_d, pts1_d, C2_d, pts2_d, P_d, N);
    cudaDeviceSynchronize();

    P_out.resize(N * 3);
    cudaMemcpy(P_out.data(), P_d, P_out_size_bytes, cudaMemcpyDeviceToHost);
    
    //frees
    cudaFree(C1_d);
    cudaFree(pts1_d);
    cudaFree(C2_d);
    cudaFree(pts2_d);
    cudaFree(P_d);

    // Reprojection error calculation would go here if implemented.
    return 0.0f; 
}