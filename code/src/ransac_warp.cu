#include "ransac_warp.cuh"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <float.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

#define WARP_ID (threadIdx.x / WARP_SIZE)
#define LANE_ID (threadIdx.x % WARP_SIZE)
#define ITER_IDX (blockIdx.x * WARPS_PER_BLOCK + WARP_ID)


__device__ float3 load_point(const float* pts, int ind) {
  float3 p;
  p.x = pts[ind * 2];
  p.y = pts[ind * 2 + 1];
  p.z = 1;
  return p;
}

__device__ int device_compute_symmetric_epipolar_dist(const float* s_F, const float* hpts1_dev, const float* hpts2_dev, int N, float threshold) {
  // int idx = threadIdx.x;
  int idx = LANE_ID;

  int local_count = 0; // # of inliers this thread encountered.

  for (int i = idx ; i < N ; i += WARP_SIZE) {
     // Get the point for image 1 and 2 this thread is in charge of.
    float3 p1 = load_point(hpts1_dev, i);
    float3 p2 = load_point(hpts2_dev, i);

    // Get Epipolar Lines l2 = F * p1.
    // l2 = [a, b, c] where ax + by + c = 0
    float3 l2;
    l2.x = s_F[0] * p1.x + s_F[1] * p1.y + s_F[2] * p1.z;
    l2.y = s_F[3] * p1.x + s_F[4] * p1.y + s_F[5] * p1.z;
    l2.z = s_F[6] * p1.x + s_F[7] * p1.y + s_F[8] * p1.z;

    // Get Epipolar Lines l1 = F.T * p2.
    float3 l1;
    l1.x = s_F[0] * p2.x + s_F[3] * p2.y + s_F[6] * p2.z;
    l1.y = s_F[1] * p2.x + s_F[4] * p2.y + s_F[7] * p2.z;
    l1.z = s_F[2] * p2.x + s_F[5] * p2.y + s_F[8] * p2.z;

    // Get numerator (epipolar constraint).
    // p2.T * F * p1, same as dot product of p2 and l2.
    float numerator = p2.x * l2.x + p2.y * l2.y + p2.z * l2.z;
    float numerator_squared = numerator * numerator;

    // Denom: squared norm of x and y of the lines.
    float denom2 = l2.x * l2.x + l2.y * l2.y;
    float denom1 = l1.x * l1.x + l1.y * l1.y;

    // Compute final epipolar dist.
    float err = (numerator_squared / denom2) + (numerator_squared / denom1);
    printf("idx %d err %f\n", idx, err);
    if (err < threshold * threshold) {
      local_count++;
    }
  }

  return local_count;
}


// ROWS: Number of rows in A
// COLS: Number of columns in A (and rows/cols of V)
// ITERS: Number of Jacobi sweeps 
template <int ROWS, int COLS, int ITERS>
__device__ void svd_jacobi_generic(float* A, float* V) {
    // Initialize V to Identity
    #pragma unroll
    for (int i = 0; i < COLS * COLS; i++) V[i] = 0.0f;
    #pragma unroll
    for (int i = 0; i < COLS; i++) V[i * COLS + i] = 1.0f;

    // Perform Sweeps
    #pragma unroll
    for (int iter = 0; iter < ITERS; iter++) {
        // Iterate over all column pairs (p < q)
        #pragma unroll
        for (int p = 0; p < COLS - 1; p++) {
            #pragma unroll
            for (int q = p + 1; q < COLS; q++) {
                
                float a = 0.0f, b = 0.0f, d = 0.0f;

                // Dot products over ROWS
                #pragma unroll
                for (int i = 0; i < ROWS; i++) {
                    float val_p = A[i * COLS + p]; 
                    float val_q = A[i * COLS + q]; 
                    a += val_p * val_p;
                    b += val_q * val_q;
                    d += val_p * val_q;
                }

                if (fabsf(d) < 1e-9f) continue; 

                // Standard Jacobi Math
                float tau = (b - a) / (2.0f * d);
                float sign = (tau >= 0.0f) ? 1.0f : -1.0f;
                float t = sign / (fabsf(tau) + sqrtf(1.0f + tau * tau));
                float c = 1.0f / sqrtf(1.0f + t * t);
                float s = c * t;

                // Apply rotation to A (ROWS)
                #pragma unroll
                for (int i = 0; i < ROWS; i++) {
                    float Ap = A[i * COLS + p];
                    float Aq = A[i * COLS + q];
                    A[i * COLS + p] = c * Ap - s * Aq;
                    A[i * COLS + q] = s * Ap + c * Aq;
                }

                // Apply rotation to V (COLS)
                #pragma unroll
                for (int i = 0; i < COLS; i++) {
                    float Vp = V[i * COLS + p];
                    float Vq = V[i * COLS + q];
                    V[i * COLS + p] = c * Vp - s * Vq;
                    V[i * COLS + q] = s * Vp + c * Vq;
                }
            }
        }
    }
}

// Extract Min Singular Vector (Templated for loop unrolling)
template <int ROWS, int COLS>
__device__ void get_min_singular_vector_generic(const float* A_final, const float* V_final, float* f_out) {
    float min_norm_sq = FLT_MAX;
    int min_col_idx = 0;

    #pragma unroll
    for (int col = 0; col < COLS; col++) {
        float norm_sq = 0.0f;
        #pragma unroll
        for (int row = 0; row < ROWS; row++) {
            float val = A_final[row * COLS + col];
            norm_sq += val * val;
        }
        
        if (norm_sq < min_norm_sq) {
            min_norm_sq = norm_sq;
            min_col_idx = col;
        }
    }

    #pragma unroll
    for (int i = 0; i < COLS; i++) {
        f_out[i] = V_final[i * COLS + min_col_idx];
    }
}


__device__ void singularize_3x3(float* F) {
    float A[9];
    float V[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) A[i] = F[i];

    // 3 Rows, 3 Cols, 4 Iterations
    svd_jacobi_generic<3, 3, 4>(A, V);

    float v_min[3];
    get_min_singular_vector_generic<3, 3>(A, V, v_min);

    // Project F
    float F_dot_v[3]; 
    F_dot_v[0] = F[0]*v_min[0] + F[1]*v_min[1] + F[2]*v_min[2];
    F_dot_v[1] = F[3]*v_min[0] + F[4]*v_min[1] + F[5]*v_min[2];
    F_dot_v[2] = F[6]*v_min[0] + F[7]*v_min[1] + F[8]*v_min[2];

    #pragma unroll
    for(int i=0; i<3; i++) {
        #pragma unroll
        for(int j=0; j<3; j++) {
            F[i*3+j] -= F_dot_v[i] * v_min[j];
        }
    }
}


// s_A is [8*9] matrix in shared mem
__device__ void device_eight_point_minimal(const float* pts1_dev, const float* pts2_dev, const size_t M, float* output_F_dev, float* s_A) {
  // int tid = threadIdx.x;

  float V[81];

  // First 8 threads for this block populates A matrix (1 thread per row).
  // Computes each of the eq given in np.stack() of the Python implementation.
  if (LANE_ID < 8) {
    // Load x,y value of points needed for this row & normalize.
    float x1 = pts1_dev[ITER_IDX * 2] / M;
    float y1 = pts1_dev[ITER_IDX * 2 + 1] / M;
    float x2 = pts2_dev[ITER_IDX * 2] / M;
    float y2 = pts2_dev[ITER_IDX * 2 + 1] / M;

    // Populate a row for the A matrix.
    s_A[LANE_ID * 9 + 0] = x2 * x1;
    s_A[LANE_ID * 9 + 1] = x2 * y1;
    s_A[LANE_ID * 9 + 2] = x2;
    s_A[LANE_ID * 9 + 3] = y2 * x1;
    s_A[LANE_ID * 9 + 4] = y2 * y1;
    s_A[LANE_ID * 9 + 5] = y2;
    s_A[LANE_ID * 9 + 6] = x1;
    s_A[LANE_ID * 9 + 7] = y1;
    s_A[LANE_ID * 9 + 8] = 1.0f;
  }
  __syncwarp();


  if (LANE_ID == 0) {
    // _, _, Vh = np.linalg.svd(A)
    svd_jacobi_generic<8, 9, 15>(s_A, V);

    // Get F_norm and singularize it (to get rank 2)
    float f[9];
    get_min_singular_vector_generic<8,9>(s_A,V,f);
    singularize_3x3(f);

    // Unscale F
    float m_inv = 1.0f / (float)M;
    float T[3] = {m_inv, m_inv, 1.0f};

    for(int r=0; r<3; r++) {
        for(int c=0; c<3; c++) {
            f[r*3+c] *= (T[r] * T[c]);
        }
    }

    float scale = (fabsf(f[8]) > 1e-10f) ? (1.0f / f[8]) : 1.0f;
    for(int i=0; i<9; i++) output_F_dev[i] = f[i] * scale;
  }
  __syncwarp();
}


// NOTE: Call with diff dims than calling ransac - only need num_blocks # of threads.
// Populates state array with the random state (will be used in ransac).
__global__ void init_rng(curandState *state, unsigned long long seed, int num_blocks) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < num_blocks) {
        // Each block needs its own random state to be independent
        curand_init(seed, id, 0, &state[id]);
    }
}


__global__ void ransac_warp_kernel(
  const float* pts1,
  const float* pts2,
  int num_points,
  float M, 
  int num_iters,
  float threshold, 
  curandState *global_states, // Each block gets their own rand state
  float* out_fund_matrix, // [num_iters * 9] dim, fund matrix for each block
  int* out_inlier_counts // [num_iters] dim, inlier count for each block
) {
  // Points for eight point algo.
  __shared__ float s_pts1[WARPS_PER_BLOCK][8 * 2];
  __shared__ float s_pts2[WARPS_PER_BLOCK][8 * 2];
  // Fundamental matrix for this block.
  __shared__ float s_F[WARPS_PER_BLOCK][9];
  // Used for reduction of inliers.
  // __shared__ int s_block_inliers;
  // Used during eight point algo.
  __shared__ float s_A[WARPS_PER_BLOCK][72]; // shape 8*9

  // int tid = threadIdx.x;
  // int bid = blockIdx.x;
    
  if (ITER_IDX >= num_iters) return;


  // if (tid == 0) {
  //   s_block_inliers = 0;
  // }
 
  // T0 randomly generates the 8 points for the algo.
  if (LANE_ID == 0) {
    curandState local_state = global_states[ITER_IDX];

    // Pick 8 unique points.
    int indices[8];
    for (int i = 0 ; i < 8 ; i++) {
      while (true) {
        // Generate num btwn 0 to N-1.
        float rand_float = curand_uniform(&local_state);
        int idx = (int) (rand_float * (num_points - 0.00001f));

        // Check uniqueness.
        bool unique = true;
        for (int k = 0 ; k < i ; k++) {
          if (indices[k] == idx) {
            unique = false;
            break;
          }
        }

        if (unique) {
          indices[i] = idx;
          break;
        }
      }

      // Load points into shared mem
      s_pts1[WARP_ID][i*2] = pts1[indices[i] * 2];
      s_pts1[WARP_ID][i*2+1] = pts1[indices[i] * 2 + 1];
      s_pts2[WARP_ID][i*2] = pts2[indices[i] * 2];
      s_pts2[WARP_ID][i*2+1] = pts2[indices[i] * 2 + 1];
    }
  }

  __syncwarp();

  // Compute fundamental matrix.
  device_eight_point_minimal(s_pts1[WARP_ID], s_pts2[WARP_ID], M, s_F[WARP_ID], s_A[WARP_ID]);
  __syncwarp();

  // Compute epipolar dist for all point pairs.
  int local_inlier_cnt = device_compute_symmetric_epipolar_dist(s_F[WARP_ID], pts1, pts2, num_points, threshold);
  printf("Before for loop local inlier count %d\n", local_inlier_cnt);
  __syncwarp();

  // Reduce and add all counts (at the end lane 0 has the reduced sum).
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      local_inlier_cnt += __shfl_down_sync(0xFFFFFFFF, local_inlier_cnt, offset);
  }
  // atomicAdd(&s_block_inliers, local_inlier_cnt); 

  // __syncwarp();

  // Write output.
  if (LANE_ID == 0) {
    printf("after accumulation, local inlier cnt is %d\n\n\n\n", local_inlier_cnt);
    out_inlier_counts[ITER_IDX] = local_inlier_cnt;
    for (int i = 0 ; i < 9 ; i++) {
      out_fund_matrix[ITER_IDX * 9 + i] = s_F[WARP_ID][i];
    }
  }
}

void cuda_ransac_warp(const std::vector<float> &pts1, const std::vector<float> &pts2, const size_t M, std::vector<float>& output_F, int num_iters, float threshold) {  
  float *pts1_dev, *pts2_dev, *out_fund_matrix_dev;
  int *out_inlier_counts_dev;
  curandState *rand_states_dev;
  size_t pts_size = pts1.size() * sizeof(float);
  size_t output_F_size = 9 * num_iters * sizeof(float);
  size_t output_inlier_cnt_size = num_iters * sizeof(int);
  size_t rand_states_size = num_iters * sizeof(curandState);

  int num_points = pts1.size() / 2;

  cudaMalloc((void**)&pts1_dev, pts_size);
  cudaMalloc((void**)&pts2_dev, pts_size);
  cudaMalloc((void**)&out_fund_matrix_dev, output_F_size);
  cudaMalloc((void**)&out_inlier_counts_dev, output_inlier_cnt_size);
  cudaMalloc((void**)&rand_states_dev, rand_states_size);

  // Copy data from Host to Device.
  cudaMemcpy(pts1_dev, pts1.data(), pts_size, cudaMemcpyHostToDevice);
  cudaMemcpy(pts2_dev, pts2.data(), pts_size, cudaMemcpyHostToDevice);

  // Initialize cudaRand state for each block in the RANSAC algo.
  int init_rng_num_blocks = (num_iters + 255) / 256;
  init_rng<<<init_rng_num_blocks, 256>>>(rand_states_dev, /* seed= */ 0, num_iters);
  cudaDeviceSynchronize();

  int threads_per_block = 256;
  int num_blocks = (num_iters + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  // Run ransac algorithm.
  ransac_warp_kernel<<<num_blocks, 256>>>(
    pts1_dev, pts2_dev, num_points, M, 
    num_iters, threshold, rand_states_dev, 
    out_fund_matrix_dev, out_inlier_counts_dev
  );
  cudaDeviceSynchronize();

  // Copy results from device to host.
  std::vector<int> inlier_counts(num_iters);
  std::vector<float> fund_matrices(num_iters * 9);
  cudaMemcpy(inlier_counts.data(), out_inlier_counts_dev, output_inlier_cnt_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fund_matrices.data(), out_fund_matrix_dev, output_F_size, cudaMemcpyDeviceToHost);

  // Find fundamental matrix with max inlier count and populate output.
  auto max_iter = std::max_element(inlier_counts.begin(), inlier_counts.end());
  int best_idx = std::distance(inlier_counts.begin(), max_iter);
  int best_count = *max_iter;
  printf("Max inliers: %d\n", best_count);
  output_F.reserve(9);
  for (int i = 0 ; i < 9 ; i++) {
    output_F[i] = fund_matrices[best_idx * 9 + i];
  }

  cudaFree(pts1_dev);
  cudaFree(pts2_dev);
  cudaFree(out_fund_matrix_dev);
  cudaFree(out_inlier_counts_dev);
  cudaFree(rand_states_dev);
}