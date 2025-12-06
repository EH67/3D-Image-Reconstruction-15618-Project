#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

#include "ransac_warp.cuh"

namespace py = pybind11;


/**
* @brief Pyhton wrapper for RANSAC algorithm in the find correspondance pipeline.
*
* @param pts1 Shape [N,2] for image 1
* @param pts2 Shape [N,2] for image 2
* @param M max of height or width of image
* @param num_iters # of iters for RANSAC
* @param threshold Used to determine inliers
*
* @returns Shape [3,3] fundamental matrix that maximizes amount of inliers
*/
py::array_t<float> py_cuda_ransac_warp(py::array_t<float> pts1, py::array_t<float> pts2, size_t M, int num_iters, float threshold) {
  // Input validation.
  if (pts1.ndim() != 2 || pts1.shape(1) != 2) {
    throw std::runtime_error("pts1 must has shape (N * 2)");
  }
  if (pts2.ndim() != 2 || pts2.shape(1) != 2) {
    throw std::runtime_error("pts2 must has shape (N * 2)");
  }
  if (pts1.shape(0) != pts2.shape(0)) {
    throw std::runtime_error("pts1 and pts2 do not have the same N");
  }

  // Convert numpy array to vectors (and collapse into 1D).
  std::vector<float> pts1_vec(pts1.data(), pts1.data() + pts1.size());
  std::vector<float> pts2_vec(pts2.data(), pts2.data() + pts2.size());
  std::vector<float> output_F_vec(9);

  cuda_ransac_warp(pts1_vec, pts2_vec, M, output_F_vec, num_iters, threshold);

  // Convert output vector to a numpy array to be returned.
  py::array_t<float> output_F_numpy(
    std::vector<size_t>{3,3}, // shape = (3,3)
    output_F_vec.data()
  );

  return output_F_numpy;
}

PYBIND11_MODULE(cuda_ransac_warp_module, m) {
    m.def("cuda_ransac_warp", &py_cuda_ransac_warp,
        "RANSAC algorithm for the correspondance pipeline implemented with warp level parallelism.");
}