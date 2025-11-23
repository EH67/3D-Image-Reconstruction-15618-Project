#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>

// Include the C++ header where the host wrapper is declared
#include "triangulate.cuh"

namespace py = pybind11;

/**
 * @brief Python wrapper for the CUDA triangulation function.
 * * Takes 4 NumPy arrays (C1, pts1, C2, pts2) and returns the triangulated 3D points P.
 * * @param C1 3x4 Camera 1 Projection Matrix.
 * @param pts1 Nx2 2D points from Image 1.
 * @param C2 3x4 Camera 2 Projection Matrix.
 * @param pts2 Nx2 2D points from Image 2.
 * @return py::array_t<float> Nx3 3D triangulated points.
 */
py::array_t<float> py_cuda_triangulate(
    py::array_t<float> C1, 
    py::array_t<float> pts1, 
    py::array_t<float> C2, 
    py::array_t<float> pts2) 
{
    // --- 1. Input Validation and Size Extraction ---
    
    // Ensure all inputs are float32 (float)
    if (C1.dtype().kind() != 'f' || pts1.dtype().kind() != 'f' || 
        C2.dtype().kind() != 'f' || pts2.dtype().kind() != 'f') {
        throw std::runtime_error("All input arrays must be of type float32 (np.float32).");
    }

    // Camera matrices must be 3x4 (12 elements)
    if (C1.size() != 12 || C2.size() != 12) {
        throw std::runtime_error("Camera matrices C1 and C2 must be 3x4 (12 elements).");
    }

    // Points arrays must have the same number of points (N)
    if (pts1.ndim() != 2 || pts2.ndim() != 2 || pts1.shape(1) != 2 || pts2.shape(1) != 2) {
         throw std::runtime_error("Point arrays must be Nx2 matrices.");
    }
    
    int N = (int)pts1.shape(0);
    if (N != (int)pts2.shape(0)) {
        throw std::runtime_error("pts1 and pts2 must have the same number of rows (N).");
    }
    
    // Check for empty input and return an empty array unambiguously
    if (N == 0) {
        // FIX: Use explicit std::vector<size_t> for dimensions
        return py::array_t<float>(std::vector<size_t>{0, 3});
    }


    // --- 2. Convert NumPy data to standard C++ vectors (required by host function) ---
    // Note: We copy the data here because the CUDA function needs the data in contiguous std::vectors.
    std::vector<float> C1_vec(C1.data(), C1.data() + C1.size());
    std::vector<float> C2_vec(C2.data(), C2.data() + C2.size());
    
    // For the points, it's efficient to just pass the raw data pointers to the vector constructors
    std::vector<float> pts1_vec(pts1.data(), pts1.data() + pts1.size());
    std::vector<float> pts2_vec(pts2.data(), pts2.data() + pts2.size());

    // --- 3. Prepare Output Vector and Call CUDA Host Function ---
    std::vector<float> P_out_vec; 
    
    // Call the CUDA host wrapper.
    float total_error = cuda_triangulate(
        C1_vec, pts1_vec, C2_vec, pts2_vec, P_out_vec
    );
    
    // Note: P_out_vec is resized and populated inside cuda_triangulate.

    // --- 4. Convert Result Vector back to NumPy Array ---
    // FIX: Use explicit std::vector<size_t> for dimensions when creating the NumPy array
    py::array_t<float> P_out_np(
        std::vector<size_t>{(size_t)N, 3}, // Dimensions (N rows, 3 columns)
        P_out_vec.data()                   // Data pointer
    );

    // Return the resulting NumPy array
    return P_out_np;
}


// --- PYBIND11 MODULE DEFINITION ---
PYBIND11_MODULE(cuda_triangulation_module, m) {
    m.doc() = "Pybind11-CUDA bindings for the Robust DLT Triangulation Kernel.";
    
    // Bind the Python wrapper function to the name 'triangulate'
    m.def("triangulate", &py_cuda_triangulate, 
        "Performs robust DLT triangulation on batches of 2D points using the GPU."
    );
}