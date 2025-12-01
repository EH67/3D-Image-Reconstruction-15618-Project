import sys
import os
import numpy as np
import time

from src.ransac import compute_symmetric_epipolar_distance

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.

# Import the Pybind11 module (assuming you name the RANSAC binding 'compute_errors')
try:
    # Assuming you compile this into 'cuda_ransac_module'
    import cuda_ransac_module 
except ImportError as e:
    print("\n--- CRITICAL ERROR: RANSAC MODULE NOT FOUND ---")
    print(f"Failed to import 'cuda_ransac_module': {e}")
    print("Please ensure the module name and path are correct.")
    sys.exit(1)
    
# --- Test Case Data Generation ---

# A large number of points to ensure the CUDA grid is fully utilized
N_points = 100000 

# Generate dummy 2D points (normalized coordinates are typically between -1 and 1)
pts1 = np.random.uniform(-1.0, 1.0, size=(N_points, 2)).astype(np.float32)
pts2 = pts1 + np.random.normal(0.0, 0.005, size=(N_points, 2)).astype(np.float32) # Add small noise

# Add 3rd dim
hpts1 = np.concatenate([pts1, np.ones([N_points, 1])], axis=1) # Nx3
hpts2 = np.concatenate([pts2, np.ones([N_points, 1])], axis=1) # Nx3

# Generate a plausible Fundamental Matrix (must be rank 2)
# F_true = U @ S @ Vh, where S[2,2] = 0
U_rand = np.linalg.svd(np.random.rand(3, 3))[0]
V_rand = np.linalg.svd(np.random.rand(3, 3))[0]
S_diag = np.diag([0.1, 0.05, 0.0]) # Two non-zero singular values, one zero
F_model = U_rand @ S_diag @ V_rand
F_model = F_model.astype(np.float32)

print(f"--- RANSAC Error Calculation Verification (N={N_points}) ---")

# 1. Compute Ground Truth (NumPy)
start_time_np = time.time()
errors_cpu = compute_symmetric_epipolar_distance(F_model, hpts1, hpts2)
end_time_np = time.time()
print(f"NumPy (Ground Truth) Time: {(end_time_np - start_time_np) * 1000:.3f} ms")


# 2. Compute CUDA Result
try:
    N = pts1.shape[0]
    # hpts1_flat = np.concatenate([pts1, np.ones([N, 1], dtype=np.float32)], axis=1).flatten()
    # hpts2_flat = np.concatenate([pts2, np.ones([N, 1], dtype=np.float32)], axis=1).flatten()
    # F_flat = F_model.flatten()

    start_time_gpu = time.time()
    errors_gpu_flat = cuda_ransac_module.cuda_compute_symmetric_epipolar_dist(F_model, pts1, pts2)
    end_time_gpu = time.time()
    print(f"CUDA (GPU Result) Time: {(end_time_gpu - start_time_gpu) * 1000:.3f} ms")
    
    # Convert flat output back to standard array
    errors_gpu = np.array(errors_gpu_flat).reshape(-1)

    # 3. Verification
    
    # Calculate the mean absolute difference (MAD) between CPU and GPU
    mad = np.mean(np.abs(errors_gpu - errors_cpu))
    
    # Check if the results are numerically close
    tolerance = 1e-5 
    
    print("-" * 50)
    print(f"Mean Absolute Difference (MAD) between CPU and GPU: {mad:.8f}")
    
    # Display a sample for inspection
    sample_indices = [0, 1,2,3,1000, N_points - 1]
    print("\nSample Error Values:")
    for i in sample_indices:
        print(f"Index {i:6d} | CPU: {errors_cpu[i]:.8f} | GPU: {errors_gpu[i]:.8f} | Diff: {(errors_gpu[i] - errors_cpu[i]):.8e}")

    if mad < tolerance:
        print(f"\nVerification successful: MAD is within the tolerance of {tolerance:.8f}.")
    else:
        print(f"\nVerification FAILED: MAD is {mad:.8f}, which is above the tolerance of {tolerance:.8f}.")

except Exception as e:
    print(f"\nAn error occurred during CUDA execution: {e}")