#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

#include "ransac.cuh"

namespace py = pybind11;


/**
* @brief Python wrapper for computing epipolar dist.
*
* @param F 3*3 fundamental matrix
* @param hpts1 N*2 homogeneous points from image 1
* @param hpts2 N*2 homogeneous points from image 2
* 
* Unlike the python version where hpts1 and hpts2 are N*3 in which dim 3 was filled with 1s so that
* numpy can perform matrix ops and shape matches, for the cuda version we are keeping it as N*2
* and only keeping the point information.
*
* @return a n 
*/
py::array_t<float> py_cuda_compute_symmetric_epipolar_dist(
    py::array_t<float> F, 
    py::array_t<float> hpts1, 
    py::array_t<float> hpts2) {
      // Input validation - all arrays are type float & shape checks.
      if (F.dtype().kind() != 'f' || hpts1.dtype().kind() != 'f' || hpts2.dtype().kind() != 'f') {
        throw std::runtime_error("All input arrays must be of type float32 (np.float32).");
      }
      if (F.size() != 9 || F.ndim() != 2 || F.shape(0) != 3 || F.shape(1) != 3) {
        throw std::runtime_error("Fundamental matrix must be 3*3.");
      }
      if (hpts1.ndim() != 2 || hpts1.shape(1) != 2) {
        throw std::runtime_error("hpts1 must has shape (N * 2)");
      }
      if (hpts2.ndim() != 2 || hpts2.shape(1) != 2) {
        throw std::runtime_error("hpts2 must has shape (N * 2)");
      }
      if (hpts1.shape(0) != hpts2.shape(0)) {
        throw std::runtime_error("hpts1 and hpts2 do not have the same N");
      }

      printf("Passed all input validation checks in py_cuda_compute_symmetric_epopilara_dist\n");

      // Convert numpy array to vectors (and collapse all into 1D vector).
      size_t output_size = hpts1.shape(0);
      std::vector<float> F_vec(F.data(), F.data() + F.size());
      std::vector<float> hpts1_vec(hpts1.data(), hpts1.data() + hpts1.size());
      std::vector<float> hpts2_vec(hpts2.data(), hpts2.data() + hpts2.size());
      std::vector<float> output_vec(output_size);

      // Call wrapper to CUDA function.
      cuda_compute_symmetric_epipolar_dist(F_vec, hpts1_vec, hpts2_vec, output_vec);

      // Convert output vector to a numpy array to be returned.
      py::array_t<float> output_numpy(
        std::vector<size_t>{output_size}, // shape = (output_size,)
        output_vec.data()
      );

      return output_numpy;
    }

/**
* @brief Python wrapper for eight point minimal algorithm (w/o nonlinear refinement at the end).
*
* @param pts1 Shape [8,2] for image 1
* @param pts2 Shape [8,2] for image 2
* @param M Max of width / height of image
*
* @returns a 3*3 Fundamental matrix
*/
py::array_t<float> py_cuda_eight_point_minimal(py::array_t<float> pts1, py::array_t<float> pts2, size_t M) {
  // Input validation.
  if (pts1.dtype().kind() != 'f' || pts2.dtype().kind() != 'f') {
    throw std::runtime_error("All input arrays must be of type float32 (np.float32).");
  }
  if (pts1.ndim() != 2 || pts1.shape(0) != 8 || pts1.shape(1) != 2) {
    throw std::runtime_error("Pts1 must have shape [8,2]. Ensure the array is transposed correctly.");
  }
  if (pts2.ndim() != 2 || pts2.shape(0) != 8 || pts2.shape(1) != 2) {
    throw std::runtime_error("Pts2 must have shape [8,2]. Ensure the array is transposed correctly.");
  }

  // Convert numpy array to vectors (and collapse all into 1D vector).
  std::vector<float> pts1_vec(pts1.data(), pts1.data() + pts1.size());
  std::vector<float> pts2_vec(pts2.data(), pts2.data() + pts2.size());
  std::vector<float> output_F_vec(9);
  printf("inside py cuda eight point minimal, about to call cuda\n");
  cuda_eight_point_minimal(pts1_vec, pts2_vec, M, output_F_vec);
  printf("finished callign cuda eight point minimal\n");

  // Convert output vector to a numpy array to be returned.
  py::array_t<float> output_F_numpy(
    std::vector<size_t>{3,3}, // shape = (3,3)
    output_F_vec.data()
  );

  return output_F_numpy;
}

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
py::array_t<float> py_cuda_ransac(py::array_t<float> pts1, py::array_t<float> pts2, size_t M, int num_iters, float threshold) {
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

  cuda_ransac(pts1_vec, pts2_vec, M, output_F_vec, num_iters, threshold);

  // Convert output vector to a numpy array to be returned.
  py::array_t<float> output_F_numpy(
    std::vector<size_t>{3,3}, // shape = (3,3)
    output_F_vec.data()
  );

  return output_F_numpy;
}

PYBIND11_MODULE(cuda_ransac_module, m) {
    m.doc() = "Pybind11-CUDA bindings for the Ransac kernel.";
    
    // Bind the Python wrapper function to the name 'compute_symmetric_epipolar_distance'
    m.def("cuda_compute_symmetric_epipolar_dist", &py_cuda_compute_symmetric_epipolar_dist, 
        "Compute the epipolar distance for all homogeneous points using the GPU."
    );

    m.def("cuda_eight_point_minimal", &py_cuda_eight_point_minimal,
        "Eight point algorithm without the refinement.");

    m.def("cuda_ransac", &py_cuda_ransac,
        "RANSAC algorithm for the correspondance pipeline.");
}