# Sample Main
# FIX: The first line "this i" seems like an accidental artifact. Removing it.

import sys
import os
import numpy as np
import time

# --- Setup for loading the compiled CUDA module ---

current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate common CMake output paths relative to this script:
# The current directory is usually '.../code', so root_dir is '.../project'
root_dir = os.path.dirname(current_dir)

# Location 1: Top-level build directory (usually where 'cmake' was run)
# Based on your path: .../3D-Image-Reconstruction-15618-Project/build
build_dir_global = os.path.join(root_dir, 'build')

# Location 2: Local build directory (e.g., inside the code folder)
build_dir_local = os.path.join(current_dir, 'build')

# Prioritize the global build location, where CMake usually places the module
if os.path.exists(build_dir_global):
    sys.path.insert(0, build_dir_global)
    print(f"Adding global build path to sys.path: {build_dir_global}")
elif os.path.exists(build_dir_local):
    sys.path.insert(0, build_dir_local)
    print(f"Adding local build path to sys.path: {build_dir_local}")
else:
    # Fallback to local path, but we'll let the user know what's expected
    # The file should still be successfully imported if the user ran the build step correctly.
    sys.path.insert(0, build_dir_local)
    print(f"Assuming module is compiled into: {build_dir_local}")


# Import the Pybind11 module defined by PYBIND11_MODULE
try:
    # Attempt to import the module
    import cuda_triangulation_module 
except ImportError as e:
    # If the import fails, print detailed debugging information.
    print("\n--- CRITICAL ERROR: MODULE NOT FOUND ---")
    print(f"Failed to import 'cuda_triangulation_module': {e}")
    print("Please ensure the following:")
    print("1. You ran the full CMake/Make process successfully.")
    print("2. The compiled file is named 'cuda_triangulation_module.so' or 'cuda_triangulation_module.pyd'.")
    print("3. That file exists in one of the paths searched:")
    for path in sys.path:
        if 'build' in path:
            print(f"- {path}")
    sys.exit(1)
    
# --- Initialize data for Triangulation (3 pairs of points, matching C++ test case) ---

# Camera matrices (3x4)
N_points = 3
C1 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=np.float32)
C2 = np.array([
    [1.0, 0.0, 0.0, 1.0], 
    [0.0, 1.0, 0.0, 0.0], 
    [0.0, 0.0, 1.0, 0.0]  
], dtype=np.float32)

# 2D points (Nx2) - Using the perfect case for a known result: P_true=(1, 1, 1)
pts1 = np.array([
    [1.0, 1.0],     # Perfect Case
    [0.01, -0.01],  # Noisy Case
    [-1.0, 2.0]     # Mock Case
], dtype=np.float32)

pts2 = np.array([
    [2.0, 1.0],     # Perfect Case projection
    [-0.01, 0.01],  # Noisy Case projection
    [0.0, 2.0]      # Mock Case projection
], dtype=np.float32)

# --- Call C++/Cuda function ---
print("Launching CUDA Triangulation kernel via Pybind11...")
start_time = time.time()
# Call the new function 'triangulate' from the new module
P_gpu = cuda_triangulation_module.triangulate(C1, pts1, C2, pts2)
end_time = time.time()

# 3. Correct Verification: We check the shape and print the result.
# The expected result for the first point (1, 1, 1) should be close to (1.0, 1.0, 1.0)

print("\n--- Results ---")
print(f"CUDA execution time: {(end_time - start_time) * 1000:.3f} ms for {N_points} points.")
print(f"Output shape P_gpu: {P_gpu.shape}")
print("P_gpu (First 3D Point):", P_gpu[0])

# Expected result for the perfect case P[0] = (1, 1, 1)
expected_P0 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
if np.allclose(P_gpu[0], expected_P0, atol=1e-3):
    print(f"\nVerification successful for P[0]: Result {P_gpu[0]} matches expected {expected_P0}.")
else:
    print(f"\nVerification failed for P[0]: Result {P_gpu[0]} does NOT match expected {expected_P0}.")