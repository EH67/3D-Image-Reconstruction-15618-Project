#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

#include "triangulate.cuh"

namespace py = pybind11;

py::array_t<float> py_cuda_triangulate(
    py::array_t<float, py::array::c_style | py::array::forcecast> C1, 
    py::array_t<float, py::array::c_style | py::array::forcecast> pts1, 
    py::array_t<float, py::array::c_style | py::array::forcecast> C2, 
    py::array_t<float, py::array::c_style | py::array::forcecast> pts2) 
{

    
    int N = pts1.shape(0);

    
    if (N == 0) {
        return py::array_t<float>(std::vector<ssize_t>{0, 3});
    }
    
    // Convert to vectors
    std::vector<float> C1_vec(12);
    std::vector<float> C2_vec(12);
    std::vector<float> pts1_vec(N * 2);
    std::vector<float> pts2_vec(N * 2);
    
    auto C1_buf = C1.unchecked<2>();
    auto C2_buf = C2.unchecked<2>();
    auto pts1_buf = pts1.unchecked<2>();
    auto pts2_buf = pts2.unchecked<2>();
    
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) {
            C1_vec[i*4+j] = C1_buf(i, j);
            C2_vec[i*4+j] = C2_buf(i, j);
        }
    
    for (int i = 0; i < N; i++) {
        pts1_vec[i*2+0] = pts1_buf(i, 0);
        pts1_vec[i*2+1] = pts1_buf(i, 1);
        pts2_vec[i*2+0] = pts2_buf(i, 0);
        pts2_vec[i*2+1] = pts2_buf(i, 1);
    }
    
    // Call CUDA
    std::vector<float> result_vec = cuda_triangulate(C1_vec, pts1_vec, C2_vec, pts2_vec);
    
    // Convert back to NumPy
    py::array_t<float> result(std::vector<ssize_t>{N, 3});
    auto result_buf = result.mutable_unchecked<2>();
    
    for (int i = 0; i < N; i++) {
        result_buf(i, 0) = result_vec[i*3+0];
        result_buf(i, 1) = result_vec[i*3+1];
        result_buf(i, 2) = result_vec[i*3+2];
    }
    
    return result;
}

PYBIND11_MODULE(cuda_triangulation_module, m) {
    m.def("triangulate", &py_cuda_triangulate);
}
