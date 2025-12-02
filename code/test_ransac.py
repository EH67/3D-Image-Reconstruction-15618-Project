import sys
import os
import numpy as np
import time
from src.ransac import ransac_fundamental_matrix
import random

# --- 1. Setup Module Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)

try:
    import cuda_ransac_module
    print("Successfully imported cuda_ransac_module")
except ImportError as e:
    print("\n--- CRITICAL ERROR: RANSAC MODULE NOT FOUND ---")
    print(f"Failed to import 'cuda_ransac_module': {e}")
    sys.exit(1)

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
    return np.sum(errors < threshold)

# --- 3. Test Execution ---

def run_correctness_test():
    print("\n" + "="*50)
    print(" PART 1: CORRECTNESS CHECK (N=8, Iters=1)")
    print("="*50)
    
    M = 1000.0
    NUM_ITERS = 1  
    THRESHOLD = 3.0 # Large threshold to ensure floating point diffs don't disqualify inliers
    
    np.random.seed(random.randint(1,10000000)) 
    
    # Generate exactly 8 points (Deterministic Sample)
    pts1_raw = np.random.uniform(0, M, size=(8, 2)).astype(np.float32)
    pts2_raw = pts1_raw + np.random.uniform(-10, 10, size=(8, 2)).astype(np.float32)
    
    # Python
    F_python, mask = ransac_fundamental_matrix(pts1_raw, pts2_raw, M, NUM_ITERS, THRESHOLD)
    
    # CUDA
    try:
        F_cuda = cuda_ransac_module.cuda_ransac(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
    except Exception as e:
        print(f"CUDA Failed: {e}")
        return

    # Print Matrices
    print("-" * 30)
    print("Python F Matrix:")
    print(F_python)
    print("\nCUDA F Matrix:")
    print(F_cuda)
    print("-" * 30)

    # Comparison of Matrix Values
    diff_pos = np.abs(F_python - F_cuda)
    diff_neg = np.abs(F_python + F_cuda)
    actual_diff = min(np.max(diff_pos), np.max(diff_neg))
    
    print(f"Max Matrix Element Diff: {actual_diff:.8f}")

    # Comparison of Inliers
    inliers_py = count_inliers(F_python, pts1_raw, pts2_raw, THRESHOLD)
    if inliers_py != np.sum(mask): # sanity check to make sure count inliers is accurate
      print("ERROR IN COUNTING INLIERS IN TEST: inliers_py is", inliers_py, "but the algo returns ", np.sum(mask))
      exit(1)
    else:
      print("py", inliers_py, "sum", np.sum(mask))
    inliers_cuda = count_inliers(F_cuda, pts1_raw, pts2_raw, THRESHOLD)

    print(f"Inliers Python: {inliers_py} / 8")
    print(f"Inliers CUDA:   {inliers_cuda} / 8")

    if actual_diff < 1e-3 and inliers_py == inliers_cuda:
        print(">> SUCCESS: Implementations match on deterministic input.")
    else:
        print(">> WARNING: Implementations diverge.")
        if inliers_py != inliers_cuda:
            print("   -> Inlier count mismatch!")

def run_benchmark():
    print("\n" + "="*50)
    print(" PART 2: SCALE BENCHMARK (N=5000, Iters=2000)")
    print("="*50)
    
    M = 1000.0
    NUM_ITERS = 2000  # Realistic number of RANSAC iterations
    THRESHOLD = 5.0
    N_POINTS = 5000   # Realistic number of feature points
    
    np.random.seed(random.randint(0,100000000))
    
    # Generate synthetic data with 50% outliers
    # Inliers:
    pts1_in = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_in = pts1_in + np.random.normal(0, 1.0, size=(N_POINTS // 2, 2)).astype(np.float32) # Small noise
    
    # Outliers:
    pts1_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32)
    pts2_out = np.random.uniform(0, M, size=(N_POINTS // 2, 2)).astype(np.float32) # Totally random matches
    
    pts1 = np.vstack([pts1_in, pts1_out])
    pts2 = np.vstack([pts2_in, pts2_out])
    
    # Shuffle
    perm = np.random.permutation(N_POINTS)
    pts1 = pts1[perm]
    pts2 = pts2[perm]
    
    # --- Python Benchmark ---
    print(f"Running Python RANSAC ({NUM_ITERS} iters on {N_POINTS} points)...")
    start_py = time.time()
    F_py, mask = ransac_fundamental_matrix(pts1, pts2, M, NUM_ITERS, THRESHOLD)
    time_py = (time.time() - start_py) * 1000.0
    
    inliers_py = count_inliers(F_py, pts1, pts2, THRESHOLD)
    if inliers_py != np.sum(mask): # sanity check to make sure count inliers is accurate
      print("ERROR IN COUNTING INLIERS IN TEST: inliers_py is", inliers_py, "but the algo returns ", np.sum(mask))
      exit(1)
    else:
      print("py", inliers_py, "sum", np.sum(mask))
    print(f"Python Time: {time_py:.2f} ms | Inliers found: {inliers_py}")

    # --- CUDA Benchmark ---
    print(f"Running CUDA RANSAC ({NUM_ITERS} iters on {N_POINTS} points)...")
    try:
        # Warmup (optional, to init context)
        # cuda_ransac_module.cuda_ransac(pts1[:8], pts2[:8], int(M), 1, THRESHOLD)
        
        start_cuda = time.time()
        F_cuda = cuda_ransac_module.cuda_ransac(pts1, pts2, int(M), NUM_ITERS, THRESHOLD)
        time_cuda = (time.time() - start_cuda) * 1000.0
        
        inliers_cuda = count_inliers(F_cuda, pts1, pts2, THRESHOLD)
        print(f"CUDA Time:   {time_cuda:.2f} ms | Inliers found: {inliers_cuda}")
        
        # --- Speedup Stats ---
        print("-" * 30)
        print(f"Speedup: {time_py / time_cuda:.2f}x")
        
        # --- Quality Sanity Check ---
        # CUDA random sampling might find a slightly better or worse model, 
        # but it should be in the same ballpark (e.g., within 5-10% of ground truth inliers)
        # Ground truth is roughly N_POINTS // 2 (2500)
        if inliers_cuda > (N_POINTS * 0.4): 
            print(">> SUCCESS: CUDA found a valid model (high inlier count).")
        else:
            print(">> WARNING: CUDA inlier count is suspiciously low. Check RANSAC logic.")
            
    except Exception as e:
        print(f"CUDA Failed: {e}")

if __name__ == "__main__":
    run_correctness_test()
    run_benchmark()