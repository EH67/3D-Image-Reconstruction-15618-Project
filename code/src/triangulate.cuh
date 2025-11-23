#include <vector>
#include <string>
#include <algorithm>

/**
 * @brief Performs triangulation of 2D image points to 3D world points on the GPU.
 *
 * This function handles the host side memory allocation, copying data to the GPU,
 * computing 3D points using DLT, and copying the results back to the host.
 *
 * @param C1    3x4 Camera Projection Matrix for the first image.
 * @param pts1  Nx2 matrix of 2D points from the first image.
 * @param C2    3x4 Camera Projection Matrix for the second image.
 * @param pts2  Nx2 matrix of 2D points from the second image.
 * @param P_out Vector of computed 3D points
 * @return      Reprojection error (currently a placeholder 0.0f, will implement if there is time).
 */
float cuda_triangulate(const std::vector<float>& C1, const std::vector<float>& pts1,
                       const std::vector<float>& C2, const std::vector<float>& pts2, 
                       std::vector<float>& P_out);