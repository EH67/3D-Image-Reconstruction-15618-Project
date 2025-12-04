import sys
import os
import numpy as np
import time
import cv2
from src.find_M2 import findM2_cpu, findM2_gpu


# ==========================================
# 4. SYNTHETIC DATA GENERATION
# ==========================================

def generate_synthetic_data(n_points=1000):
    """
    Generates 3D points and projects them into two cameras to create
    perfect correspondences and a ground truth Fundamental Matrix.
    """
    # 1. Setup Camera 1 (Identity)
    K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]], dtype=np.float32)
    M1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ M1
    
    # 2. Setup Camera 2 (Rotation + Translation)
    # Rotate 10 degrees around Y axis, Translate x=+1, z=0
    theta = np.radians(10)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    t = np.array([[-1.0], [0.2], [0.1]]) # Parallax is needed
    M2 = np.hstack((R, t))
    P2 = K @ M2
    
    # 3. Generate 3D points in front of cameras
    # X range: -2 to 2, Y range: -2 to 2, Z range: 5 to 15
    points_3d = np.random.rand(n_points, 3) * np.array([4, 4, 10]) + np.array([-2, -2, 5])
    
    # 4. Project to 2D
    def project(P, points):
        points_h = np.hstack((points, np.ones((len(points), 1))))
        proj = (P @ points_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return proj.astype(np.float32)
    
    pts1 = project(P1, points_3d)
    pts2 = project(P2, points_3d)
    
    # 5. Compute Ground Truth Fundamental Matrix
    # F = K^-T * [t]x * R * K^-1
    tx = np.array([
        [0, -t[2,0], t[1,0]],
        [t[2,0], 0, -t[0,0]],
        [-t[1,0], t[0,0], 0]
    ])
    E = tx @ R
    K_inv = np.linalg.inv(K)
    F = K_inv.T @ E @ K_inv
    
    intrinsics = {'K1': K, 'K2': K}
    
    return pts1, pts2, F, intrinsics, points_3d, M2

# ==========================================
# 5. MAIN BENCHMARK
# ==========================================

if __name__ == "__main__":
    
    N_POINTS = 1000 
    print(f"Generating synthetic data with {N_POINTS} points...")
    pts1, pts2, F, intrinsics, gt_3d, gt_M2 = generate_synthetic_data(N_POINTS)
    
    print("-" * 60)
    print("1. Running CPU Version...")
    start_cpu = time.perf_counter()
    M2_cpu, C2_cpu, P_cpu = findM2_cpu(F, pts1, pts2, intrinsics)
    end_cpu = time.perf_counter()
    time_cpu = end_cpu - start_cpu
    print(f"   CPU Time: {time_cpu:.4f} seconds")
    
    print("-" * 60)
    print("2. Running GPU Version...")
    # Warmup
    findM2_gpu(F, pts1[:100], pts2[:100], intrinsics)
    
    start_gpu = time.perf_counter()
    M2_gpu, C2_gpu, P_gpu = findM2_gpu(F, pts1, pts2, intrinsics)
    end_gpu = time.perf_counter()
    time_gpu = end_gpu - start_gpu
    print(f"   GPU Time: {time_gpu:.4f} seconds")
    
    print("-" * 60)
    print("3. Comparison & Correctness")
    
    # Check if GPU result is None
    if M2_gpu is None:
        print("   [FAIL] GPU returned None!")
        sys.exit(1)

    # Compare 3D reconstruction center of mass (to handle scale/ordering issues)
    # Since findM2 returns normalized matrices, exact values might differ by scale,
    # but the structure of the cloud should be the same.
    center_cpu = np.mean(P_cpu, axis=0)
    center_gpu = np.mean(P_gpu, axis=0)
    
    dist_diff = np.linalg.norm(center_cpu - center_gpu)
    print(f"   3D Center Difference (CPU vs GPU): {dist_diff:.6f}")
    
    if dist_diff < 0.1:
        print("   [PASS] CPU and GPU results match.")
    else:
        print("   [WARN] CPU and GPU results diverge.")

    # Speedup
    print("-" * 60)
    print(f"Speedup: {time_cpu / time_gpu:.2f}x")
    print("-" * 60)