"""
Benchmark for triangulation on different number of points.
"""

import sys
import os
import numpy as np
import time

# --- Sequential CPU Implementation (Reference) ---

def triangulate_cpu(C1, pts1, C2, pts2):
  '''
  Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
  Input:  C1, the 3x4 camera matrix
          pts1, the Nx2 matrix with the 2D image coordinates per row
          C2, the 3x4 camera matrix
          pts2, the Nx2 matrix with the 2D image coordinates per row
  Output: P, the Nx3 matrix with the corresponding 3D points per row
          err, the reprojection error.

  Hints:
  (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
  (2) Solve for the least square solution using np.linalg.svd
  (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from
      homogeneous coordinates to non-homogeneous ones)
  (4) Keep track of the 3D points and projection error, and continue to next point
  (5) You do not need to follow the exact procedure above.
  '''

  # ----- TODO -----
  ### BEGIN SOLUTION
  N = pts1.shape[0]
  P = np.zeros((N, 3))
  total_error = 0.0

  for i in range(N):
    p1 = pts1[i, :]
    p2 = pts2[i, :]

    A = np.zeros((4, 4))
    A[0, :] = p1[0] * C1[2, :] - C1[0, :]
    A[1, :] = p1[1] * C1[2, :] - C1[1, :]
    A[2, :] = p2[0] * C2[2, :] - C2[0, :]
    A[3, :] = p2[1] * C2[2, :] - C2[1, :]

    U, S, Vh = np.linalg.svd(A)
    P_homo = Vh[-1, :]

    P_non_homo = P_homo[0:3] / P_homo[3]
    P[i, :] = P_non_homo

    p1_proj_homo = C1 @ P_homo
    p1_proj = p1_proj_homo[0:2] / p1_proj_homo[2]
    p2_proj_homo = C2 @ P_homo
    p2_proj = p2_proj_homo[0:2] / p2_proj_homo[2]

    #error
    err1 = np.sum((p1_proj - p1)**2)
    err2 = np.sum((p2_proj - p2)**2)
    total_error += (err1 + err2)

  err = total_error

  ### END SOLUTION

  return P, err

# --- CUDA Module Setup ---

current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory (where main.py and cuda_triangulation_module.so now reside) 
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Adding current directory to sys.path: {current_dir}")

# Import the Pybind11 module 
try:
    import cuda_triangulation_module 
except ImportError as e:
    print("\n--- CRITICAL ERROR: MODULE NOT FOUND ---")
    print(f"Failed to import 'cuda_triangulation_module': {e}")
    sys.exit(1)
    
# --- Data Generation Function ---

def generate_test_data(N_points, C1, C2):
    """Generates ground truth points and corresponding noisy 2D projections."""
    # Generate random points in a positive Z region (in front of the camera)
    TRUE_POINTS = np.random.uniform(low=-5.0, high=5.0, size=(N_points, 3)).astype(np.float32)
    TRUE_POINTS[:, 2] = np.abs(TRUE_POINTS[:, 2]) + 1.0 # Ensure Z is positive (depth)
    
    # Calculate the true projections
    pts1_true = TRUE_POINTS[:, :2] / TRUE_POINTS[:, 2:3] # p1 = [X/Z, Y/Z]
    pts2_true_num = TRUE_POINTS[:, :2] + C2[0, 3]       # Numerator [X+tX, Y+tY]
    pts2_true = pts2_true_num / TRUE_POINTS[:, 2:3]     # p2 = [(X+1)/Z, Y/Z]

    # Add realistic noise
    noise = np.random.normal(0.0, 0.001, size=pts1_true.shape).astype(np.float32)
    pts1 = pts1_true + noise
    pts2 = pts2_true + noise

    return TRUE_POINTS, pts1, pts2

if __name__ == "__main__":
    # --- Camera matrices (3x4) ---
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


    # =========================================================================
    # 1. ACCURACY TEST (Small Batch - N=100)
    # =========================================================================
    print("=" * 60)
    print("1. ACCURACY TEST (N=100)")
    print("=" * 60)

    N_accuracy = 100
    TRUE_POINTS_ACC, pts1_acc, pts2_acc = generate_test_data(N_accuracy, C1, C2)

    # Run CPU Reference
    start_time_cpu = time.time()
    P_cpu, err_cpu = triangulate_cpu(C1, pts1_acc, C2, pts2_acc)
    time_cpu_acc = time.time() - start_time_cpu

    # Run CUDA SVD  Solver
    start_time_cuda = time.time()
    P_cuda = cuda_triangulation_module.triangulate(C1, pts1_acc, C2, pts2_acc)
    time_cuda_acc = time.time() - start_time_cuda

    print(f'ref time: {time_cpu_acc}, cuda time: {time_cuda_acc}')


    # --- Accuracy Verification ---

    # Calculate the mean absolute distance between the CUDA result and the TRUE point
    mae_cuda = np.mean(np.abs(P_cuda - TRUE_POINTS_ACC))

    # Calculate the mean absolute distance between the CUDA result and the CPU SVD result
    # This checks the numerical consistency of the cuSOLVER SVD solver vs. the industry-standard SVD.
    max_diff_vs_cpu = np.max(np.abs(P_cuda - P_cpu))

    print(f"Accuracy Test Completed for N={N_accuracy} points.")
    print(f"-> MAE (CUDA vs. True P): {mae_cuda:.6f}")
    print(f"-> Max Absolute Difference (CUDA vs. CPU SVD): {max_diff_vs_cpu:.6f}")

    # Target check: Max difference should be very small (e.g., less than 0.01 units).
    if max_diff_vs_cpu < 0.01:
        print("-> RESULT: CUDA SVD is numerically consistent with CPU SVD (acceptable difference).")
    else:
        print("-> WARNING: High numerical divergence between CUDA SVD and CPU SVD.")


    # =========================================================================
    # 2. PERFORMANCE TEST (Large Batch - N=1,000,000)
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. PERFORMANCE TEST (N=1,000,000)")
    print("=" * 60)

    N_perf = 1000000
    TRUE_POINTS_PERF, pts1_perf, pts2_perf = generate_test_data(N_perf, C1, C2)

    # --- CPU Time ---
    start_time_cpu = time.time()
    P_cpu_perf, err_cpu_perf = triangulate_cpu(C1, pts1_perf, C2, pts2_perf)
    time_cpu_perf = time.time() - start_time_cpu

    # --- CUDA Time ---
    start_time_cuda = time.time()
    P_cuda_perf = cuda_triangulation_module.triangulate(C1, pts1_perf, C2, pts2_perf)
    time_cuda_perf = time.time() - start_time_cuda


    # --- Performance Summary ---
    print(f"Processed {N_perf} points.")
    print(f"   CPU (NumPy SVD) Time: {time_cpu_perf * 1000:.3f} ms")
    print(f"   CUDA (cuSOLVER) Time: {time_cuda_perf * 1000:.3f} ms")

    if time_cuda_perf > 0 and time_cpu_perf > 0:
        speedup = time_cpu_perf / time_cuda_perf
        print(f"\n   SPEEDUP FACTOR: {speedup:.2f}x")
        if speedup > 2.0:
            print("-> RESULT: Significant parallel speedup achieved!")
        else:
            print("-> RESULT: Minimal speedup. Consider optimizing CUDA thread scheduling.")
    else:
        print("Performance calculation failed (timing error).")
