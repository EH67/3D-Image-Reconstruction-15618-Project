#include <vector>
#include <string>
#include <algorithm>

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
