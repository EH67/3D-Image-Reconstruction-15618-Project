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
* @param hpts1 N*3 homogeneous points from image 1
* @param hpts2 N*3 homogeneous points from image 2
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
      if (hpts1.ndim() != 2 || hpts1.shape(1) != 3) {
        throw std::runtime_error("hpts1 must has shape (N * 3)");
      }
      if (hpts2.ndim() != 2 || hpts2.shape(1) != 3) {
        throw std::runtime_error("hpts2 must has shape (N * 3)");
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
      std::vector<float> output_vec;

      // Call wrapper to CUDA function.
      printf("About to call cuda_compute_symmetric_epipolar_dist\n");
      cuda_compute_symmetric_epipolar_dist(F_vec, hpts1_vec, hpts2_vec, output_vec);


      // Convert output vector to a numpy array to be returned.
      py::array_t<float> output_numpy(
        std::vector<size_t>{output_size}, // shape = (output_size,)
        output_vec
      );

      return output_numpy;
    }

PYBIND11_MODULE(cuda_ransac_module, m) {
    m.doc() = "Pybind11-CUDA bindings for the Ransac kernel.";
    
    // Bind the Python wrapper function to the name 'compute_symmetric_epipolar_distance'
    m.def("cuda_compute_symmetric_epipolar_dist", &py_cuda_compute_symmetric_epipolar_dist, 
        "Compute the epipolar distance for all homogeneous points using the GPU."
    );
}