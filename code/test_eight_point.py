import sys
import os
import numpy as np
import time

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

# --- 2. Helper & Python Implementation ---

def _singularize(F):
    '''
    Enforces rank 2 on the Fundamental matrix.
    '''
    U, S, Vh = np.linalg.svd(F)
    S[-1] = 0 # Enforce rank 2 by zeroing the smallest singular value
    return U @ np.diag(S) @ Vh

def eightpoint_minimal(pts1, pts2, M):
    '''
    Python implementation with added print statements for debugging comparison.
    '''
    print(f"Python: pts1.shape {pts1.shape}")
    print(f"Python: M {M}")
    
    N = pts1.shape[0]
    T = np.array([[1.0/M, 0,     0],
                  [0,     1.0/M, 0],
                  [0,     0,     1]])

    pts1_norm = pts1 / M
    pts2_norm = pts2 / M

    x1, y1 = pts1_norm[:, 0], pts1_norm[:, 1]
    x2, y2 = pts2_norm[:, 0], pts2_norm[:, 1]
    
    # Construct A matrix
    # [x2x1, x2y1, x2, y2x1, y2y1, y2, x1, y1, 1]
    A = np.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(N)
    ], axis=1)
    
    print(f"Python: A shape {A.shape}")
    
    # --- ADDED FOR COMPARISON ---
    print("\n--- PYTHON A MATRIX START ---")
    # Setting print options to ensure precision matches typical C++ float output
    np.set_printoptions(precision=6, suppress=True, linewidth=200)
    print(A)
    print("--- PYTHON A MATRIX END ---\n")
    # ----------------------------

    _, _, Vh = np.linalg.svd(A)
    f = Vh[-1, :]
    F_norm = f.reshape(3, 3)
    
    F_norm_rank2 = _singularize(F_norm)

    F_unscaled = T.T @ F_norm_rank2 @ T
    
    # Normalize by the last element if not zero
    if F_unscaled[2, 2] != 0:
        F_unscaled = F_unscaled / F_unscaled[2, 2]
    
    return F_unscaled

# --- 3. Test Execution ---

def main():
    print("Initializing Comparison Test...")
    
    # Constants
    M = 1000.0
    
    # Generate exactly 8 random points (required by your CUDA wrapper)
    # Using a fixed seed so you can compare multiple runs if needed
    np.random.seed(42) 
    
    # Generate raw points (0 to M)
    pts1_raw = np.random.uniform(0, M, size=(8, 2)).astype(np.float32)
    # Make pts2 slightly offset from pts1
    pts2_raw = pts1_raw + np.random.uniform(-10, 10, size=(8, 2)).astype(np.float32)
    
    print("\n" + "="*50)
    print(" 1. RUNNING PYTHON IMPLEMENTATION")
    print("="*50)
    
    start_py = time.time()
    F_python = eightpoint_minimal(pts1_raw, pts2_raw, M)
    end_py = time.time()
    
    print(f"Python Result F (Top 2 rows):\n{F_python[:2, :]}")
    
    print("\n" + "="*50)
    print(" 2. RUNNING CUDA IMPLEMENTATION")
    print("="*50)
    
    # Ensure inputs are contiguous and float32 for PyBind
    pts1_cuda = np.ascontiguousarray(pts1_raw, dtype=np.float32)
    pts2_cuda = np.ascontiguousarray(pts2_raw, dtype=np.float32)
    
    try:
        start_cuda = time.time()
        # This function call should trigger the prints inside your CUDA C++ code
        F_cuda = cuda_ransac_module.cuda_eight_point_minimal(pts1_cuda, pts2_cuda, int(M))
        end_cuda = time.time()
        
        print(f"\nCUDA Result F (Top 2 rows):\n{F_cuda[:2, :]}")
        
    except Exception as e:
        print(f"CUDA execution failed: {e}")
        return

    # --- Final Comparison of Resulting F ---
    print("\n" + "="*50)
    print(" COMPARISON OF FINAL F MATRIX")
    print("="*50)
    
    diff = np.abs(F_python - F_cuda)
    max_diff = np.max(diff)
    
    print(f"Max absolute difference in F: {max_diff:.8f}")
    
    if max_diff < 1e-4:
        print(">> SUCCESS: Python and CUDA results match closely.")
    else:
        print(">> WARNING: Results diverge significantly. Check your A matrix construction and SVD.")

if __name__ == "__main__":
    main()