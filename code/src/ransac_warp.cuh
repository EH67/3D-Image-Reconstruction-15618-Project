#include <vector>
#include <string>
#include <algorithm>

// /**
// * @brief Calculate epipolar distance of all the potential corresponding points btwn the images.
// *
// * @param F The 3 * 3 fundamental matrix represented as a 1D array.
// * @param hpts1 The homogeneous points from image 1. 1D array of N*3 elements (convert to float3).
// * @param hpts2 The homogeneous points from image 2. 1D array of N*3 elements (convert to float3).
// * @return A size N 1D vector containing the epipolar dist across all homogeneous points.
// */
// void cuda_compute_symmetric_epipolar_dist(const std::vector<float>& F, 
//                                           const std::vector<float>& hpts1,
//                                           const std::vector<float>& hpts2,
//                                           std::vector<float>& output);

// void cuda_eight_point_minimal(const std::vector<float>& pts1, const std::vector<float>&pts2, const size_t M, std::vector<float>& output_F);

/**
* @brief Ransac algorithm for finding the best fundamental matrix.
*
* @param pts1 2D array of [N,2] dim, containing all points from image 1.
* @param pts2 2D array of [N,2] dim, containing all points from image 2.
* @param M Max between width and height of the image.
* @param output_F 2D array of [3,3] dim as the output fundamental matrix.
* @param num_iters Number of iterations for RANSAC.
* @param threshold Threshold for determining inliers.
*/
void cuda_ransac_warp(const std::vector<float> &pts1, const std::vector<float> &pts2, const size_t M, std::vector<float>& output_F, int num_iters, float threshold);
