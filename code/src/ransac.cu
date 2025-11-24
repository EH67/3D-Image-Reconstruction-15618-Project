#include "ransac.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <float.h>

__device__ float3 load_point(const float* pts, int ind) {
  float3 p;
  p.x = pts[ind * 3];
  p.y = pts[ind * 3 + 1];
  p.z = pts[ind * 3 + 2];
  return p;
}

__global__ void kernel_compute_symmetric_epipolar_dist(const float* F_dev, const float* hpts1_dev, const float* hpts2_dev, float* output_dev, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Cooperatively load fundamental matrix into shared memory.
  __shared__ float s_F[9];
  printf("After declaring shared mem for fundamental matrix\n");
  if (threadIdx.x < 9) {
    s_F[threadIdx.x] = F_dev[threadIdx.x];
  }
  __syncthreads();
  printf("After cooperative loading\n");

  if (idx < N) {
     // Get the point for image 1 and 2 this thread is in charge of.
    float3 p1 = load_point(hpts1_dev, idx);
    float3 p2 = load_point(hpts2_dev, idx);

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
    output_dev[idx] = err;
  }
}

void cuda_compute_symmetric_epipolar_dist(const std::vector<float>& F, 
                                          const std::vector<float>& hpts1,
                                          const std::vector<float>& hpts2,
                                          std::vector<float>& output) {
  float *F_dev, *hpts1_dev, *hpts2_dev, *output_dev;
  size_t F_size = 9 * sizeof(float);
  size_t num_points = hpts1.size() / 3;
  size_t hpts_size = hpts1.size() * sizeof(float);
  size_t output_size = num_points * sizeof(float);
  

  // Allocate device memory.
  cudaMalloc((void**)&F_dev, 9 * sizeof(float));
  cudaMalloc((void**)&hpts1_dev, hpts_size);
  cudaMalloc((void**)&hpts2_dev, hpts_size);
  cudaMalloc((void**)&output, output_size);

  // Copy data from Host to Device.
  cudaMemcpy(F_dev, F.data(), F_size, cudaMemcpyHostToDevice);
  cudaMemcpy(hpts1_dev, hpts1.data(), hpts_size, cudaMemcpyHostToDevice);
  cudaMemcpy(hpts2_dev, hpts2.data(), hpts_size, cudaMemcpyHostToDevice);

  // Configure and launch kernel.
  // 1 thread = 1 homogeneous point pair btwn the two images.
  int threadsPerBlock = 256;
  int numBlocks = (num_points + threadsPerBlock - 1) / threadsPerBlock;
  printf("About to call kernel func for epipolar dist\n");
  kernel_compute_symmetric_epipolar_dist<<<numBlocks, threadsPerBlock, sizeof(float) * 9>>>(F_dev, hpts1_dev, hpts2_dev, output_dev, num_points);


  // Copy result from Device to Host.
  cudaMemcpy(output.data(), output_dev, output_size, cudaMemcpyDeviceToHost);
                                            
}
