#include <vector>
#include <string>
#include <algorithm>

/**
* @brief Calculate epipolar distance of all the potential corresponding points btwn the images.
*
* @param F The 3 * 3 fundamental matrix represented as a 1D array.
* @param hpts1 The homogeneous points from image 1. 1D array of N*3 elements (convert to float3).
* @param hpts2 The homogeneous points from image 2. 1D array of N*3 elements (convert to float3).
* @return A size N 1D vector containing the epipolar dist across all homogeneous points.
*/
void cuda_compute_symmetric_epipolar_dist(const std::vector<float>& F, 
                                          const std::vector<float>& hpts1,
                                          const std::vector<float>& hpts2,
                                          std::vector<float>& output);
                                  