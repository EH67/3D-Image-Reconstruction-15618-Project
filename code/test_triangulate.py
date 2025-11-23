# Sample Main

import sys
import os
import numpy as np
import time

# --- Setup for loading the compiled CUDA module ---

current_dir = os.path.dirname(os.path.abspath(__file__))

# FIX: Add the current directory (where main.py and cuda_triangulation_module.so now reside) 
# to sys.path directly. This is the most reliable way given your build location.
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Adding current directory to sys.path: {current_dir}")

# Import the Pybind11 module defined by PYBIND11_MODULE
try:
    # Attempt to import the module
    import cuda_triangulation_module 
except ImportError as e:
    # If the import fails, print detailed debugging information.
    print("\n--- CRITICAL ERROR: MODULE NOT FOUND ---")
    print(f"Failed to import 'cuda_triangulation_module': {e}")
    print("Please ensure the following:")
    print("1. You ran the full CMake/Make (or ninja) process successfully.")
    print("2. The compiled file 'cuda_triangulation_module.so' exists in this directory:")
    print(f"- {current_dir}")
    sys.exit(1)
    
# --- Test Case Definition (10 Points) ---

# Camera matrices (3x4)
C1 = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=np.float32)
C2 = np.array([
    [1.0, 0.0, 0.0, 1.0],  # Camera 2 translated 1 unit along X-axis
    [0.0, 1.0, 0.0, 0.0], 
    [0.0, 0.0, 1.0, 0.0]  
], dtype=np.float32)


# List of true 3D points (X, Y, Z) and corresponding 2D projections (p1, p2)
# p_2 = C2 * P_homo, where P_homo = [X, Y, Z, 1]. C2 * P_homo = [X+1, Y, Z]
# p_2_non_homo = [(X+1)/Z, Y/Z]
#--------------------------------------------------------------------------------------------------
# | True P (X, Y, Z) | p1 (C1: X/Z, Y/Z) | p2 (C2: (X+1)/Z, Y/Z) | Description
#--------------------------------------------------------------------------------------------------
TRUE_POINTS = np.array([
    [ 1.0,  1.0,  1.0], # 0. Standard/Perfect Case (Should yield [1, 1, 1])
    [ 0.0,  0.0,  5.0], # 1. Far Point (Deep Z)
    [ 5.0,  0.0,  1.0], # 2. Wide X (Large image coordinate)
    [-3.0,  2.0,  1.0], # 3. Negative X
    [ 0.0, 10.0, 10.0], # 4. Tall Y (Near top/bottom image edge)
    [ 0.0,  0.0,  0.1], # 5. Very Close Point (Shallow Z)
    [ 1.0,  1.0,  2.0], # 6. Mid-range depth
    [-2.0, -2.0,  2.0], # 7. Negative X, Y, Mid-range depth
    [ 0.5,  0.5,  1.0], # 8. Fractional coordinates
    [ 2.5,  0.5,  5.0]  # 9. Mixed wide and deep
], dtype=np.float32)

N_points = TRUE_POINTS.shape[0]

# Calculate the projections and add minimal noise for robustness testing
pts1_true = TRUE_POINTS[:, :2] / TRUE_POINTS[:, 2:3] # p1 = [X/Z, Y/Z]
pts2_true_num = TRUE_POINTS[:, :2] + [1.0, 0.0]     # Numerator [X+1, Y]
pts2_true = pts2_true_num / TRUE_POINTS[:, 2:3]     # p2 = [(X+1)/Z, Y/Z]

# Add a tiny bit of noise to simulate real input (0.001 pixel noise)
noise = np.random.normal(0.0, 0.001, size=pts1_true.shape).astype(np.float32)
pts1 = pts1_true + noise
pts2 = pts2_true + noise


# --- Call C++/Cuda function ---
print(f"Launching CUDA Triangulation kernel via Pybind11 for N={N_points} points...")
start_time = time.time()
# Call the new function 'triangulate' from the new module
P_gpu = cuda_triangulation_module.triangulate(C1, pts1, C2, pts2)
end_time = time.time()

# --- Verification and Output ---

print("\n--- Triangulation Results ---")
print(f"CUDA execution time: {(end_time - start_time) * 1000:.3f} ms for {N_points} points.")
print(f"Output shape P_gpu: {P_gpu.shape}\n")

# Calculate the mean absolute error (MAE) between the GPU result and the known truth
mae = np.mean(np.abs(P_gpu - TRUE_POINTS))

print(f"Mean Absolute Error (MAE) vs. True 3D Points: {mae:.5f}")

# Display a subset of results for inspection
print("\nSample Results (GPU Output vs. True Point):")
print("-" * 50)
for i in range(N_points):
    P_out_i = P_gpu[i]
    P_true_i = TRUE_POINTS[i]
    
    # Check error relative to the true point
    error_i = np.linalg.norm(P_out_i - P_true_i)
    
    # Print the first few and last point
    if i < 5 or i == N_points - 1:
        print(f"Index {i:2d} | True P: ({P_true_i[0]:.3f}, {P_true_i[1]:.3f}, {P_true_i[2]:.3f}) | Solved P: ({P_out_i[0]:.3f}, {P_out_i[1]:.3f}, {P_out_i[2]:.3f}) | Error: {error_i:.5f}")
    elif i == 5:
        print("...")

# Final verification check
tolerance = 0.01 # 1 cm tolerance in world space
if mae < tolerance:
    print(f"\nVerification successful: MAE is {mae:.5f}, which is below the tolerance of {tolerance:.3f}.")
else:
    print(f"\nVerification failed: MAE is {mae:.5f}, which is above the tolerance of {tolerance:.3f}.")
