import sys
import os
import numpy as np
import time
import cv2

from src.correspondence import get_correspondences_cpu, get_correspondences_gpu
from src.find_M2 import findM2_cpu, findM2_gpu 

# --- Path Setup ---
# Ensure we can import from src/ and build/
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, current_dir) # To allow importing find_m2 if it's in the root





def compare_matrices_frobenius(M1, M2, name="Matrix"):
    """
    Compares two matrices using Frobenius norm difference.
    Handles scale and sign ambiguity.
    """
    if M1 is None or M2 is None:
        print(f">> {name}: One of the matrices is None. Cannot compare.")
        return float('inf')

    # Normalize
    M1_norm = M1 / np.linalg.norm(M1)
    M2_norm = M2 / np.linalg.norm(M2)

    # Handle sign ambiguity (flipped direction)
    if np.sum(M1_norm * M2_norm) < 0:
        M2_norm = -M2_norm

    diff = np.linalg.norm(M1_norm - M2_norm)
    
    status = "SUCCESS" if diff < 0.2 else "WARNING"
    if diff > 1.0: status = "FAILURE"
    
    print(f">> {name} Similarity (Frobenius Diff): {diff:.4f} [{status}]")
    return diff

def run_pipeline(mode, im1, im2, M, intrinsics):
    """
    Runs the full pipeline (Correspondence + RANSAC + M2 + Triangulation)
    mode: 'cpu' or 'gpu'
    """
    print(f"[{mode.upper()}] Starting Pipeline...")
    start_time = time.perf_counter()

    # 1. Get Correspondences & Fundamental Matrix
    if mode == 'cpu':
        pts1, pts2, F = get_correspondences_cpu(im1, im2, M)
    else:
        pts1, pts2, F = get_correspondences_gpu(im1, im2, M)

    t1 = time.perf_counter()
    corr_time = t1 - start_time
    
    if F is None:
        print(f"[{mode.upper()}] Failed to find Fundamental Matrix.")
        return None, None, None, 0

    # 2. Find M2 (Pose) and Triangulate Points
    if mode == 'cpu':
        M2, C2, P = findM2_cpu(F, pts1, pts2, intrinsics)
    else:
        M2, C2, P = findM2_gpu(F, pts1, pts2, intrinsics)

    end_time = time.perf_counter()
    total_time = end_time - start_time
    m2_time = end_time - t1

    num_points = P.shape[0] if P is not None else 0
    
    print(f"[{mode.upper()}] Complete.")
    print(f"    - Correspondence + RANSAC Time: {corr_time:.4f}s")
    print(f"    - FindM2 + Triangulation Time:  {m2_time:.4f}s")
    print(f"    - Total Time:                   {total_time:.4f}s")
    print(f"    - 3D Points Generated:          {num_points}")
    
    return F, M2, P, total_time

if __name__ == '__main__':
    # --- 1. Load Data ---
    IMAGE_DIR = os.path.abspath(os.path.join(current_dir, 'images'))
    IM1_PATH = os.path.join(IMAGE_DIR, 'bag3.jpg')
    IM2_PATH = os.path.join(IMAGE_DIR, 'bag4.jpg')

    im1 = cv2.imread(IM1_PATH)
    im2 = cv2.imread(IM2_PATH)

    if im1 is None or im2 is None:
        print("Error: Could not load images. Check paths.")
        sys.exit(1)

    h, w, _ = im1.shape
    M_scale = np.max([w, h])
    
    # Mock Intrinsics
    focal_length = w * 1.2
    cx, cy = w / 2.0, h / 2.0
    K_mock = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    intrinsics = {'K1': K_mock, 'K2': K_mock}

    print(f"Processing Resolution: {w}x{h}")
    print("-" * 60)

    # --- 2. Run CPU ---
    F_cpu, M2_cpu, P_cpu, time_cpu = run_pipeline('cpu', im1, im2, M_scale, intrinsics)
    print("-" * 60)

    # --- 3. Run GPU ---
    F_gpu, M2_gpu, P_gpu, time_gpu = run_pipeline('gpu', im1, im2, M_scale, intrinsics)
    print("-" * 60)

    # --- 4. Comparison & Analysis ---
    print("SUMMARY RESULTS")
    
    # Speedup
    if time_gpu > 0:
        speedup = time_cpu / time_gpu
        print(f"Total Speedup: {speedup:.2f}x")
    else:
        print("Total Speedup: N/A (GPU Error)")

    # Correctness - Fundamental Matrix
    compare_matrices_frobenius(F_cpu, F_gpu, "Fundamental Matrix (F)")

    # Correctness - M2 Matrix
    # M2 is 3x4. We check if the camera pose is recovered similarly.
    compare_matrices_frobenius(M2_cpu, M2_gpu, "Camera Matrix (M2)")

    # Point Cloud Consistency
    if P_cpu is not None and P_gpu is not None:
        # We can't compare points row-by-row because RANSAC is random (different inliers).
        # But we can compare the scale/bounds of the reconstruction.
        cpu_mean = np.mean(P_cpu, axis=0)
        gpu_mean = np.mean(P_gpu, axis=0)
        dist = np.linalg.norm(cpu_mean - gpu_mean)
        print(f">> Point Cloud Centroid Distance: {dist:.4f}")
        
        # Ratio of points found
        ratio = len(P_gpu) / len(P_cpu)
        print(f">> Point Count Ratio (GPU/CPU): {ratio:.2f}")
    else:
        print(">> Cannot compare Point Clouds (one is None).")