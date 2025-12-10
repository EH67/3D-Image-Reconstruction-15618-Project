"""
Test correctness and benchmark for the whole flow (RANSAC + find_M2).
"""

import sys
import os
import numpy as np
import cv2
import time
import random
from src.ransac import ransac_fundamental_matrix
from src.correspondence import extract_features, compute_matches, filter_matches, run_ransac_cpu, run_ransac_gpu,run_ransac_warp_gpu
from src.find_M2 import findM2_cpu, findM2_gpu

# --- 1. Setup Module Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)

import cuda_ransac_module
import cuda_ransac_warp_module

EXPORT_OUTPUT = False
RESIZE = False

def export_visualization_data(filename, img1, img2, pts1, pts2, P):
    """
    Saves images, inlier coordinates, and 3D points to a .npz file.
    """
    print(f"Exporting data to {filename}...")
    # Ensure points are numpy arrays
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    if P is None:
        P = np.array([])
    
    # Save compressed to save space
    np.savez_compressed(
        filename,
        img1=img1,
        img2=img2,
        pts1=pts1,
        pts2=pts2,
        P=P
    )
    print("Export complete.")

#resize function to reduce number of pixels to run correspondence on
def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None: return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

# --- Helper for Verification ---
def count_inliers(F, pts1, pts2, threshold):
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
        print("F cuda block", F_cuda_block)
    except Exception as e:
        print(f"CUDA Block Failed: {e}")
        F_cuda_block = None

    # 3. CUDA Warp
    try:
        F_cuda_warp = cuda_ransac_warp_module.cuda_ransac_warp(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
        print("F cuda warp", F_cuda_warp)
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

# --- Helper for generating synthetic data ---
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

# --- Benchmarks ---

def run_iter_scaling_benchmark():
    print("\n" + "="*80)
    print(f"{'PART 2: ITERATION SCALING (Full Pipeline: RANSAC + FindM2)':^80}")
    print(f"{'Fixed N=5000':^80}")
    print("="*80)
    
    ITER_CONFIGS = [1000, 5000, 10000, 50000, 100000, 200000, 400000]
    M = 1000.0
    THRESHOLD = 5.0
    N_POINTS = 5000   
    
    # Mock Intrinsics for synthetic data
    K_mock = np.array([[M, 0, M/2], [0, M, M/2], [0, 0, 1]])
    intrinsics = {'K1': K_mock, 'K2': K_mock}
    
    np.random.seed(123)
    pts1, pts2 = generate_data(N_POINTS, M)

    # Warmup
    try:
        run_ransac_gpu(pts1[:8], pts2[:8], M, 1, THRESHOLD)
        run_ransac_warp_gpu(pts1[:8], pts2[:8], M, 1, THRESHOLD)
    except: pass

    print(f"{'Iters':<10} | {'Py (ms)':<10} | {'Block (ms)':<10} | {'Warp (ms)':<10} | {'Speedup (vs Py)':<15} | {'Speedup (vs Blk)':<15}")
    print("-" * 80)

    for num_iters in ITER_CONFIGS:
        # 1. Python (Skip if too slow)
        if num_iters > 2000000:
            t_py = -1.0
        else:
            s = time.perf_counter()
            # A. RANSAC
            F, mask = run_ransac_cpu(pts1, pts2, M, num_iters, THRESHOLD)
            # B. Find M2
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_cpu(F, pts1[mask], pts2[mask], intrinsics)
            t_py = (time.perf_counter() - s) * 1000.0

        # 2. CUDA Block
        try:
            s = time.perf_counter()
            F, mask = run_ransac_gpu(pts1, pts2, M, num_iters, THRESHOLD)
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_gpu(F, pts1[mask], pts2[mask], intrinsics)
            t_block = (time.perf_counter() - s) * 1000.0
        except: t_block = -1.0

        # 3. CUDA Warp
        try:
            s = time.perf_counter()
            F, mask = run_ransac_warp_gpu(pts1, pts2, M, num_iters, THRESHOLD)
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_gpu(F, pts1[mask], pts2[mask], intrinsics)
            t_warp = (time.perf_counter() - s) * 1000.0
        except: t_warp = -1.0

        str_py = f"{t_py:.2f}" if t_py > 0 else "SKIP"
        str_blk = f"{t_block:.2f}" if t_block > 0 else "ERR"
        str_warp = f"{t_warp:.2f}" if t_warp > 0 else "ERR"
        
        speedup_py = f"{t_py / t_warp:.1f}x" if (t_warp > 0 and t_py > 0) else "-"
        speedup_blk = f"{t_block / t_warp:.1f}x" if (t_warp > 0 and t_block > 0) else "-"

        print(f"{num_iters:<10} | {str_py:<10} | {str_blk:<10} | {str_warp:<10} | {speedup_py:<15} | {speedup_blk:<15}")

def run_point_scaling_benchmark():
    print("\n" + "="*80)
    print(f"{'PART 3: POINT SCALING (Full Pipeline: RANSAC + FindM2)':^80}")
    print(f"{'Fixed Iters=5000':^80}")
    print("="*80)
    
    # We test small to very large point clouds
    POINT_CONFIGS = [1000, 5000, 10000, 50000, 100000, 200000, 400000]
    M = 1000.0
    THRESHOLD = 5.0
    NUM_ITERS = 5000
    
    # Mock Intrinsics
    K_mock = np.array([[M, 0, M/2], [0, M, M/2], [0, 0, 1]])
    intrinsics = {'K1': K_mock, 'K2': K_mock}
    
    # Warmup
    try:
        p_w, _ = generate_data(100, M)
        run_ransac_gpu(p_w, p_w, M, 1, THRESHOLD)
        run_ransac_warp_gpu(p_w, p_w, M, 1, THRESHOLD)
    except: pass

    print(f"{'Points (N)':<10} | {'Py (ms)':<10} | {'Block (ms)':<10} | {'Warp (ms)':<10} | {'Speedup (vs Py)':<15} | {'Speedup (vs Blk)':<15}")
    print("-" * 80)

    for n_points in POINT_CONFIGS:
        
        # Generate new data for this size
        np.random.seed(123)
        pts1, pts2 = generate_data(n_points, M)

        # 1. Python (Skip if too slow)
        if n_points > 1000000:
            t_py = -1.0
        else:
            s = time.perf_counter()
            F, mask = run_ransac_cpu(pts1, pts2, M, NUM_ITERS, THRESHOLD)
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_cpu(F, pts1[mask], pts2[mask], intrinsics)
            t_py = (time.perf_counter() - s) * 1000.0

        # 2. CUDA Block
        try:
            s = time.perf_counter()
            F, mask = run_ransac_gpu(pts1, pts2, M, NUM_ITERS, THRESHOLD)
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_gpu(F, pts1[mask], pts2[mask], intrinsics)
            t_block = (time.perf_counter() - s) * 1000.0
        except: t_block = -1.0

        # 3. CUDA Warp
        try:
            s = time.perf_counter()
            F, mask = run_ransac_warp_gpu(pts1, pts2, M, NUM_ITERS, THRESHOLD)
            if F is not None:
                mask = mask.ravel().astype(bool)
                findM2_gpu(F, pts1[mask], pts2[mask], intrinsics)
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