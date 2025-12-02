import sys
import os
import numpy as np
import time
from src.ransac import ransac_fundamental_matrix

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


# --- 3. Test Execution ---

def main():
    print("Initializing Deterministic RANSAC Test...")
    
    # Constants
    M = 1000.0
    NUM_ITERS = 1  
    THRESHOLD = 10.0 
    
    # --- Generate Data ---
    np.random.seed(0) 
    
    # Generate random 8 points
    pts1_raw = np.random.uniform(0, M, size=(8, 2)).astype(np.float32)
    # Make pts2 related to pts1 via some random F (approx) + noise
    pts2_raw = pts1_raw + np.random.uniform(-5, 5, size=(8, 2)).astype(np.float32)
    
    print("\n" + "="*50)
    print(" 1. RUNNING PYTHON IMPLEMENTATION (EightPoint)")
    print("="*50)
    
    # Start Timing Python
    start_py = time.time()
    
    # For N=8, RANSAC is mathematically equivalent to just running EightPoint once.
    F_python, _ = ransac_fundamental_matrix(pts1_raw, pts2_raw, M, NUM_ITERS, THRESHOLD)
    
    end_py = time.time()
    time_py = (end_py - start_py) * 1000.0
    
    print(f"Python Result F:\n{F_python}")
    print(f"Python Execution Time: {time_py:.4f} ms")
    
    print("\n" + "="*50)
    print(" 2. RUNNING CUDA IMPLEMENTATION (RANSAC 1 Iter)")
    print("="*50)
    
    try:
        # Start Timing CUDA
        # Note: The C++ wrapper includes cudaDeviceSynchronize(), so this timing 
        # correctly captures the GPU execution time + memory transfer overhead.
        start_cuda = time.time()
        
        # We call the full RANSAC pipeline, but with 1 iter and 8 points.
        F_cuda = cuda_ransac_module.cuda_ransac(pts1_raw, pts2_raw, int(M), NUM_ITERS, THRESHOLD)
        
        end_cuda = time.time()
        time_cuda = (end_cuda - start_cuda) * 1000.0
        
        print(f"CUDA Result F:\n{F_cuda}")
        print(f"CUDA Execution Time:   {time_cuda:.4f} ms")
        
    except Exception as e:
        print(f"CUDA execution failed: {e}")
        return

    # --- Final Comparison ---
    print("\n" + "="*50)
    print(" COMPARISON")
    print("="*50)
    
    # Handle Sign Ambiguity: SVD can return v or -v. Both result in valid F.
    # We check if F_py is close to F_cuda OR -F_cuda
    diff_pos = np.abs(F_python - F_cuda)
    diff_neg = np.abs(F_python + F_cuda)
    
    min_diff_pos = np.max(diff_pos)
    min_diff_neg = np.max(diff_neg)
    
    actual_diff = min(min_diff_pos, min_diff_neg)
    
    print(f"Max absolute difference in F: {actual_diff:.8f}")
    
    # We allow a slightly looser tolerance here because:
    # 1. Python uses float64 (double) SVD.
    # 2. CUDA uses float32 (single) Jacobi SVD.
    tolerance = 1e-3
    
    if actual_diff < tolerance:
        print(f">> SUCCESS: Python and CUDA results match (within {tolerance}).")
    else:
        print(f">> WARNING: Results diverge. Diff: {actual_diff:.8f}")
        print("Note: Small differences are expected due to float32 vs float64 precision.")
        
    print("-" * 30)
    if time_cuda < time_py:
        print(f"Speedup: {time_py / time_cuda:.2f}x faster")
    else:
        print(f"Slowdown: {time_cuda / time_py:.2f}x slower (Expected for N=8 due to overhead)")

if __name__ == "__main__":
    main()