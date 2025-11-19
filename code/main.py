# Sample Main

import sys
import os
import numpy as np
import time

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.

# Import any modules defined by PYBIND11_MODULE from src/bindings.cpp.
import cuda_ops_module 

# Initialize data.
N = 10**8
A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)

# Call C++/Cuda function.
print("Launching CUDA kernel via Pybind11...")
start_time = time.time()
C_gpu = cuda_ops_module.multiply_arrays(A, B)
end_time = time.time()

# Ensure correctness.
numpy_start_time = time.time()
C_cpu = A * B
numpy_end_time = time.time()
assert np.allclose(C_gpu, C_cpu)

print(f"CUDA execution time: {(end_time - start_time) * 1000:.3f} ms")
print(f"Numpy execution time: {(numpy_end_time - numpy_start_time) * 1000:.3f} ms")
print(f"Verification successful: Results match NumPy.")