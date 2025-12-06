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

# try:
#     import cuda_ransac_module
#     print("Successfully imported cuda_ransac_module")
# except ImportError as e:
#     print("\n--- CRITICAL ERROR: RANSAC MODULE NOT FOUND ---")
#     print(f"Failed to import 'cuda_ransac_module': {e}")
#     sys.exit(1)

# --- Helper for Verification ---
def count_inliers(F, pts1, pts2, threshold):
    """
    Python implementation of error checking to verify the QUALITY of the CUDA result.
    We don't expect the F matrix to be identical, but it should explain the data equally well.
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

# --- 3. Test Execution ---

def run_correctness_test():
    print("\n" + "="*60)
    print(" PART 1: CORRECTNESS CHECK (N=8, Iters=1)")
    print("="*60)
    
    M = 1000.0
    NUM_ITERS = 1  
    THRESHOLD = 3.0 
    
    np.random.seed(42) # Fixed seed for reproducibility
    
    # Generate exactly 8 points (Deterministic Sample)
    pts1_raw = np.random.uniform(0, M, size=(8, 2)).astype(np.float32)
    pts2_raw = pts1_raw + np.random.uniform(-10, 10, size=(8, 2)).astype(np.float32)
    
    # 1. Python
    F_python, mask = ransac_fundamental_matrix(pts1_raw, pts2_raw, M, NUM_ITERS, THRESHOLD)
    
    # 2. CUDA Block
    try:
        F_cuda_block = cuda_ransac_module.cuda_ransac(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
    except Exception as e:
        print(f"CUDA Block Failed: {e}")
        F_cuda_block = None

    # 3. CUDA Warp
    try:
        F_cuda_warp = cuda_ransac_warp_module.cuda_ransac_warp(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
    except AttributeError:
        print("CUDA Warp function not found. Did you bind 'cuda_ransac_warp' in bindings.cpp?")
        F_cuda_warp = None
    except Exception as e:
        print(f"CUDA Warp Failed: {e}")
        F_cuda_warp = None

    # --- Analysis ---
    print(f"{'Implementation':<20} | {'Inliers':<10} | {'F Matrix Check'}")
    print("-" * 60)

    # Check Python
    inliers_py = count_inliers(F_python, pts1_raw, pts2_raw, THRESHOLD)
    print(f"{'Python (CPU)':<20} | {inliers_py:<10} | Reference")

    # Check Block
    if F_cuda_block is not None:
        inliers_block = count_inliers(F_cuda_block, pts1_raw, pts2_raw, THRESHOLD)
        diff = np.min([np.max(np.abs(F_python - F_cuda_block)), np.max(np.abs(F_python + F_cuda_block))])
        print(f"{'CUDA (Block)':<20} | {inliers_block:<10} | Diff: {diff:.6f}")
    
    # Check Warp
    if F_cuda_warp is not None:
        inliers_warp = count_inliers(F_cuda_warp, pts1_raw, pts2_raw, THRESHOLD)
        diff = np.min([np.max(np.abs(F_python - F_cuda_warp)), np.max(np.abs(F_python + F_cuda_warp))])
        print(f"{'CUDA (Warp)':<20} | {inliers_warp:<10} | Diff: {diff:.6f}")

    print("-" * 60)
    if inliers_py == inliers_warp:
         print(">> SUCCESS: Warp implementation matches Python inliers.")
    else:
         print(">> WARNING: Warp implementation inlier count mismatch.")

def run_benchmark():
    print("\n" + "="*60)
    print(" PART 2: SCALE BENCHMARK (N=5000, Iters=5000)")
    print("="*60)
    
    M = 1000.0
    NUM_ITERS = 5000  # High iterations to stress test GPU
    THRESHOLD = 5.0
    N_POINTS = 5000   
    
    np.random.seed(123)
    
    # Generate synthetic data
    pts1_in = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_in = pts1_in + np.random.normal(0, 1.0, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts1_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    
    pts1 = np.vstack([pts1_in, pts1_out])
    pts2 = np.vstack([pts2_in, pts2_out])
    perm = np.random.permutation(N_POINTS)
    pts1 = pts1[perm]
    pts2 = pts2[perm]
    
    # 1. Python Benchmark
    print(f"1. Python RANSAC...")
    start_py = time.perf_counter()
    F_py, _ = ransac_fundamental_matrix(pts1, pts2, M, NUM_ITERS, THRESHOLD)
    time_py = (time.perf_counter() - start_py) * 1000.0
    inliers_py = count_inliers(F_py, pts1, pts2, THRESHOLD)
    print(f"   -> Time: {time_py:.2f} ms | Inliers: {inliers_py}")

    # 2. CUDA Block Benchmark
    print(f"2. CUDA Block RANSAC (Original)...")
    try:
        # Warmup
        cuda_ransac_module.cuda_ransac(pts1[:8], pts2[:8], int(M), 1, THRESHOLD)
        
        start_cuda = time.perf_counter()
        F_cuda = cuda_ransac_module.cuda_ransac(pts1, pts2, int(M), NUM_ITERS, THRESHOLD)
        time_cuda = (time.perf_counter() - start_cuda) * 1000.0
        inliers_cuda = count_inliers(F_cuda, pts1, pts2, THRESHOLD)
        print(f"   -> Time: {time_cuda:.2f} ms | Inliers: {inliers_cuda}")
    except Exception as e:
        print(f"   -> Failed: {e}")
        time_cuda = float('inf')

    # 3. CUDA Warp Benchmark
    print(f"3. CUDA Warp RANSAC (Optimized)...")
    try:
        # Warmup
        cuda_ransac_warp_module.cuda_ransac_warp(pts1[:8], pts2[:8], int(M), 1, THRESHOLD)

        start_warp = time.perf_counter()
        F_warp = cuda_ransac_warp_module.cuda_ransac_warp(pts1, pts2, int(M), NUM_ITERS, THRESHOLD)
        time_warp = (time.perf_counter() - start_warp) * 1000.0
        inliers_warp = count_inliers(F_warp, pts1, pts2, THRESHOLD)
        print(f"   -> Time: {time_warp:.2f} ms | Inliers: {inliers_warp}")
    except AttributeError:
        print("   -> 'cuda_ransac_warp' not found in module.")
        time_warp = float('inf')
    except Exception as e:
        print(f"   -> Failed: {e}")
        time_warp = float('inf')

    # --- Comparison ---
    print("-" * 60)
    print("SPEEDUP ANALYSIS:")
    if time_cuda < float('inf'):
        print(f"Block GPU vs Python:      {time_py / time_cuda:.2f}x faster")
    if time_warp < float('inf'):
        print(f"Warp GPU vs Python:       {time_py / time_warp:.2f}x faster")
    if time_cuda < float('inf') and time_warp < float('inf'):
        print(f"Warp GPU vs Block GPU:    {time_cuda / time_warp:.2f}x faster")

if __name__ == "__main__":
    run_correctness_test()
    run_benchmark()