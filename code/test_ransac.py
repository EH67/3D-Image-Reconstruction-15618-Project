"""
Benchmark for RANSAC - Returns speedup as we change # of points and # of iterations.
"""

import sys
import os
import numpy as np
import time
import random
from src.ransac import ransac_fundamental_matrix

# --- 1. Setup Module Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)

import cuda_ransac_module
import cuda_ransac_warp_module

# --- Helper for Verification ---
def count_inliers(F, pts1, pts2, threshold):
    """
    Python implementation of error checking to verify the QUALITY of the CUDA result.
    """
    if F is None: return 0
    N = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([N, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([N, 1])], axis=1)
    
    l2 = F @ hpts1.T
    l1 = F.T @ hpts2.T
    
    numerator = np.sum(hpts2.T * l2, axis=0)
    denom2 = l2[0, :]**2 + l2[1, :]**2
    denom1 = l1[0, :]**2 + l1[1, :]**2
    
    # Avoid divide by zero
    denom2[denom2 < 1e-10] = 1e-10
    denom1[denom1 < 1e-10] = 1e-10
    
    errors = (numerator**2 / denom2) + (numerator**2 / denom1)
    return np.sum(errors < threshold ** 2)

def generate_data(n_points, M=1000.0):
    """Generates synthetic stereo points with 50% outliers"""
    n_inliers = n_points // 2
    n_outliers = n_points - n_inliers
    
    pts1_in = np.random.uniform(0, M, size=(n_inliers, 2)).astype(np.float32)
    pts2_in = pts1_in + np.random.normal(0, 1.0, size=(n_inliers, 2)).astype(np.float32)
    
    pts1_out = np.random.uniform(0, M, size=(n_outliers, 2)).astype(np.float32)
    pts2_out = np.random.uniform(0, M, size=(n_outliers, 2)).astype(np.float32)
    
    pts1 = np.vstack([pts1_in, pts1_out])
    pts2 = np.vstack([pts2_in, pts2_out])
    
    perm = np.random.permutation(n_points)
    return pts1[perm], pts2[perm]

# --- 2. Test Execution ---

def run_correctness_test():
    print("\n" + "="*80)
    print(f"{'PART 1: CORRECTNESS CHECK (N=8, Iters=1)':^80}")
    print("="*80)
    
    M = 1000.0
    NUM_ITERS = 1  
    THRESHOLD = 3.0 
    
    np.random.seed(42)
    pts1_raw, pts2_raw = generate_data(8, M)
    
    # 1. Python
    F_python, mask = ransac_fundamental_matrix(pts1_raw, pts2_raw, M, NUM_ITERS, THRESHOLD)
    
    # 2. CUDA Block
    try:
        F_cuda_block = cuda_ransac_module.cuda_ransac(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
    except Exception as e:
        F_cuda_block = None

    # 3. CUDA Warp
    try:
        F_cuda_warp = cuda_ransac_warp_module.cuda_ransac_warp(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
    except Exception as e:
        F_cuda_warp = None

    # Analysis
    print(f"{'Implementation':<20} | {'Inliers':<10} | {'Status'}")
    print("-" * 80)

    inliers_py = count_inliers(F_python, pts1_raw, pts2_raw, THRESHOLD)
    print(f"{'Python (CPU)':<20} | {inliers_py:<10} | Reference")

    if F_cuda_block is not None:
        inliers_block = count_inliers(F_cuda_block, pts1_raw, pts2_raw, THRESHOLD)
        print(f"{'CUDA (Block)':<20} | {inliers_block:<10} | {'MATCH' if inliers_block == inliers_py else 'MISMATCH'}")
    
    if F_cuda_warp is not None:
        inliers_warp = count_inliers(F_cuda_warp, pts1_raw, pts2_raw, THRESHOLD)
        print(f"{'CUDA (Warp)':<20} | {inliers_warp:<10} | {'MATCH' if inliers_warp == inliers_py else 'MISMATCH'}")
    print("-" * 80)

def run_iter_scaling_benchmark():
    print("\n" + "="*80)
    print(f"{'PART 2: ITER SCALING (Fixed Points=5000)':^80}")
    print("="*80)
    
    # --- Configuration ---
    # We will test these iteration counts
    ITER_CONFIGS = [1000, 5000, 10000, 50000, 100000, 200000, 400000]
    
    M = 1000.0
    THRESHOLD = 5.0
    N_POINTS = 5000   
    
    # --- Generate Data (Once) ---
    # We use the same dataset for all iterations to ensure fair comparison
    np.random.seed(123)
    pts1_in = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_in = pts1_in + np.random.normal(0, 1.0, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts1_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    
    pts1 = np.vstack([pts1_in, pts1_out])
    pts2 = np.vstack([pts2_in, pts2_out])
    perm = np.random.permutation(N_POINTS)
    pts1 = pts1[perm]
    pts2 = pts2[perm]

    # --- Warmup GPUs ---
    # Run a small dummy kernel to initialize CUDA context overhead before timing
    try:
        cuda_ransac_module.cuda_ransac(pts1[:8], pts2[:8], int(M), 1, THRESHOLD)
        cuda_ransac_warp_module.cuda_ransac_warp(pts1[:8], pts2[:8], int(M), 1, THRESHOLD)
    except:
        pass

    # --- Table Header ---
    # Cols: Iters | Py Time | Block Time | Warp Time | Speedup (Py/Warp) | Speedup (Block/Warp)
    print(f"{'Iters':<10} | {'Py (ms)':<10} | {'Block (ms)':<10} | {'Warp (ms)':<10} | {'Speedup (vs Py)':<15} | {'Speedup (vs Blk)':<15}")
    print("-" * 80)

    for num_iters in ITER_CONFIGS:
        
        # 1. Python
        # Skip Python for very large iters if it takes too long (optional check)
        if num_iters > 10000000:
            t_py = -1.0 # Skip
        else:
            s = time.perf_counter()
            _, _ = ransac_fundamental_matrix(pts1, pts2, M, num_iters, THRESHOLD)
            t_py = (time.perf_counter() - s) * 1000.0

        # 2. CUDA Block
        try:
            s = time.perf_counter()
            _ = cuda_ransac_module.cuda_ransac(pts1, pts2, int(M), num_iters, THRESHOLD)
            t_block = (time.perf_counter() - s) * 1000.0
        except:
            t_block = -1.0

        # 3. CUDA Warp
        try:
            s = time.perf_counter()
            _ = cuda_ransac_warp_module.cuda_ransac_warp(pts1, pts2, int(M), num_iters, THRESHOLD)
            t_warp = (time.perf_counter() - s) * 1000.0
        except:
            t_warp = -1.0

        # --- Formatting Output ---
        str_py = f"{t_py:.2f}" if t_py > 0 else "N/A"
        str_blk = f"{t_block:.2f}" if t_block > 0 else "ERR"
        str_warp = f"{t_warp:.2f}" if t_warp > 0 else "ERR"
        
        # Speedup Calcs
        if t_warp > 0 and t_py > 0:
            speedup_py = f"{t_py / t_warp:.1f}x"
        else:
            speedup_py = "-"
            
        if t_warp > 0 and t_block > 0:
            speedup_blk = f"{t_block / t_warp:.1f}x"
        else:
            speedup_blk = "-"

        print(f"{num_iters:<10} | {str_py:<10} | {str_blk:<10} | {str_warp:<10} | {speedup_py:<15} | {speedup_blk:<15}")

def run_point_scaling_benchmark():
    print("\n" + "="*80)
    print(f"{'PART 3: POINT SCALING (Fixed Iters=5000)':^80}")
    print("="*80)
    
    # We test small to very large point clouds
    POINT_CONFIGS = [1000, 5000, 10000, 50000, 100000, 200000, 400000]
    M = 1000.0
    THRESHOLD = 5.0
    NUM_ITERS = 5000
    
    # Warmup again to be safe
    try:
        pts_w, _ = generate_data(100, M)
        cuda_ransac_module.cuda_ransac(pts_w, pts_w, int(M), 1, THRESHOLD)
        cuda_ransac_warp_module.cuda_ransac_warp(pts_w, pts_w, int(M), 1, THRESHOLD)
    except: pass

    print(f"{'Points (N)':<10} | {'Py (ms)':<10} | {'Block (ms)':<10} | {'Warp (ms)':<10} | {'Speedup (vs Py)':<15} | {'Speedup (vs Blk)':<15}")
    print("-" * 80)

    for n_points in POINT_CONFIGS:
        
        # Generate new data for this size
        np.random.seed(123)
        pts1, pts2 = generate_data(n_points, M)

        # 1. Python 
        if n_points > 1000000:
            t_py = -1.0
        else:
            s = time.perf_counter()
            _, _ = ransac_fundamental_matrix(pts1, pts2, M, NUM_ITERS, THRESHOLD)
            t_py = (time.perf_counter() - s) * 1000.0

        # 2. CUDA Block
        try:
            s = time.perf_counter()
            _ = cuda_ransac_module.cuda_ransac(pts1, pts2, int(M), NUM_ITERS, THRESHOLD)
            t_block = (time.perf_counter() - s) * 1000.0
        except: t_block = -1.0

        # 3. CUDA Warp
        try:
            s = time.perf_counter()
            _ = cuda_ransac_warp_module.cuda_ransac_warp(pts1, pts2, int(M), NUM_ITERS, THRESHOLD)
            t_warp = (time.perf_counter() - s) * 1000.0
        except: t_warp = -1.0

        str_py = f"{t_py:.2f}" if t_py > 0 else "SKIP"
        str_blk = f"{t_block:.2f}" if t_block > 0 else "ERR"
        str_warp = f"{t_warp:.2f}" if t_warp > 0 else "ERR"
        
        speedup_py = f"{t_py / t_warp:.1f}x" if (t_warp > 0 and t_py > 0) else "-"
        speedup_blk = f"{t_block / t_warp:.1f}x" if (t_warp > 0 and t_block > 0) else "-"

        print(f"{n_points:<10} | {str_py:<10} | {str_blk:<10} | {str_warp:<10} | {speedup_py:<15} | {speedup_blk:<15}")

if __name__ == "__main__":
    run_correctness_test()
    run_iter_scaling_benchmark()
    run_point_scaling_benchmark()