#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept> // For std::runtime_error

#include "cuda_ops.cuh"

// Include the header where the CUDA host wrapper is declared
// Assuming your host wrapper is declared in a header file (like cuda_ops.cuh)
// but since you didn't provide one, we'll declare the C++ function here.
// NOTE: If you have a header for CUDA functions, use #include "cuda_ops.cuh" instead.
// void launch_multiply_kernel(const float* A_host, const float* B_host, float* C_host, int N);


namespace py = pybind11;

// --- FORWARD DECLARATION FOR THE PYTHON WRAPPER ---
// This tells the compiler that the function exists before the PYBIND11_MODULE calls it.
py::array_t<float> py_multiply_arrays(py::array_t<float> A, py::array_t<float> B);
// --------------------------------------------------


// --- PYTHON WRAPPER FUNCTION DEFINITION ---
// This function handles NumPy array conversion and calls the CUDA launch function.
py::array_t<float> py_multiply_arrays(py::array_t<float> A, py::array_t<float> B) {
    // 1. Check array size consistency
    if (A.size() != B.size()) {
        // Throw an exception that Python can catch
        throw std::runtime_error("Input array sizes must match.");
    }
    int N = (int)A.size();

    // 2. Create the output NumPy array that will hold the result
    auto C = py::array_t<float>(N);
    
    // 3. Get raw pointers to the data buffers (zero-copy access to NumPy data)
    const float* A_ptr = A.data();
    const float* B_ptr = B.data();
    float* C_ptr = C.mutable_data();

    // 4. Call the C++/CUDA host wrapper function
    launch_multiply_kernel(A_ptr, B_ptr, C_ptr, N);

    // 5. Return the result array
    return C;
}


// --- PYBIND11 MODULE DEFINITION ---
// The PYBIND11_MODULE macro generates the required PyInit_... function.
// We use the MODULE_NAME placeholder defined by CMake.
PYBIND11_MODULE(cuda_ops_module, m) {
    m.doc() = "Pybind11-CUDA bindings for 3D reconstruction kernels (Example: Element-wise Multiply).";
    
    // Bind the Python wrapper function to the name 'multiply_arrays'
    m.def("multiply_arrays", &py_multiply_arrays, "Multiply two NumPy arrays element-wise on the GPU.");
}