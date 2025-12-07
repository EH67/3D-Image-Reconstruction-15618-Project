import sys
import os
import numpy as np
import cv2
import time
import random
from src.ransac import ransac_fundamental_matrix
from src.correspondence import extract_features, compute_matches, filter_matches, run_ransac_cpu, run_ransac_gpu,run_ransac_warp_gpu
from src.find_M2 import findM2_cpu, findM2_gpu
import src.alignment as align
import glob

# --- 1. Setup Module Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)

import cuda_ransac_module
import cuda_ransac_warp_module

EXPORT_OUTPUT = True
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

def run_image_pipeline():
    print("\n" + "="*60)
    print(" PART 3: FULL IMAGE PIPELINE (High Accuracy / Full Res)")
    print("="*60)

    # --- Setup ---
    # Update this path to your exact folder on the GHC machine
    IMAGE_DIR = '/afs/andrew.cmu.edu/usr11/myu2/private/15418/3D-Image-Reconstruction-15618-Project/code/images' 
    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.jpg')))

    if not image_paths:
        print(f"No images found at {IMAGE_DIR}")
        return

    # --- Initialization ---
    reconstruction_data = []
    T_global = np.eye(4)
    prev_img_pts = None  
    prev_P_dense = None  

    # TUNING: For full-res images, pixel error is naturally higher. 
    # If 3.0 is too strict (rejects good matches), try 4.0 or 5.0.
    # The Colab notebook likely allows a bit more slack implicitly via SIFT quality.
    NUM_ITERS = 5000
    THRESHOLD = 3.0 

    for i in range(len(image_paths)-1):
        path1 = image_paths[i]
        path2 = image_paths[i+1]
        print(f"\n--- Pair {i}: {os.path.basename(path1)} -> {os.path.basename(path2)} ---")

        # 1. Load Full Resolution Images (DO NOT RESIZE)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        
        if img1 is None or img2 is None: continue

        h, w, _ = img1.shape
        # Log resolution to confirm we aren't accidentally downsampling
        print(f"   Resolution: {w}x{h}") 

        # 2. Dynamic Intrinsics (Calculated per image size)
        focal_length = w * 1.2
        cx = w / 2.0
        cy = h / 2.0

        K_mock = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

        # Define both formats to satisfy different function signatures
        intrinsics_dict = {'K1': K_mock, 'K2': K_mock}
        intrinsics_matrix = K_mock

        # 3. Feature Matching (Ensure src.correspondence uses SIFT!)
        print("   [Step 1] Feature Extraction...")
        kp1, des1 = extract_features(img1)
        kp2, des2 = extract_features(img2)
        matches_raw = compute_matches(des1, des2)
        pts1_cand, pts2_cand = filter_matches(matches_raw, kp1, kp2)
        print(f"   Matches found: {len(pts1_cand)}")

        if len(pts1_cand) < 8:
            print("   [Skip] Not enough matches.")
            continue

        # 4. GPU RANSAC & M2
        P_warp = None
        M2_opt = None
        pts1_in_warp = pts1_cand
        pts2_in_warp = pts2_cand

        try:
            # IMPORTANT: Ensure 'run_ransac_warp_gpu' handles large coordinates!
            # If accuracy is poor, check if this function normalizes points internally.
            F_warp, mask_gpu_warp = run_ransac_warp_gpu(pts1_cand, pts2_cand, int(max(h, w)), NUM_ITERS, THRESHOLD)
            
            # Apply Inlier Mask
            mask_gpu_warp = mask_gpu_warp.ravel().astype(bool)
            pts1_in_warp = pts1_cand[mask_gpu_warp]
            pts2_in_warp = pts2_cand[mask_gpu_warp]
            print(f"   Inliers: {len(pts1_in_warp)}")
            
            if F_warp is not None:
                # Correctly pass the Dictionary 'intrinsics_dict' here
                M2_warp, C2_warp, P_warp = findM2_gpu(F_warp, pts1_in_warp, pts2_in_warp, intrinsics_dict)
            else:
                print("   [Skip] F_warp failed.")
                continue

        except Exception as e:
            print(f"   [Error] GPU Step Failed: {e}")
            continue

        # 5. Bundle Adjustment
        # Filter to ~200 points to keep BA fast
        pts1_filt, pts2_filt, P_init_filt = align.filter_matches_BA(
            pts1_in_warp, pts2_in_warp, P_warp, img1.shape, max_points=200
        )

        M1_local = np.hstack((np.eye(3), np.zeros((3, 1))))
        try:
            # BA expects the Matrix 'intrinsics_matrix'
            M2_opt, _ = align.bundleAdjustment(
                intrinsics_matrix, M1_local, pts1_filt, 
                intrinsics_matrix, M2_warp, pts2_filt,  
                P_init_filt
            )
        except Exception as e:
            print(f"   [Warning] BA Failed: {e}. Using M2_warp.")
            M2_opt = M2_warp

        # 6. Scale Alignment
        scale = 1.0
        if prev_img_pts is not None and prev_P_dense is not None:
            # High-res points are sensitive; ensure strict alignment
            calculated_scale = align.get_scale_factor_strict(prev_img_pts, prev_P_dense, pts1_in_warp, P_warp)
            
            if calculated_scale is None:
                print(f"   [Scale] FAILED. Resetting chain.")
                # Resetting allows the next pair to start fresh rather than accumulating garbage
                T_global = np.eye(4) 
                prev_img_pts = pts2_in_warp
                prev_P_dense = P_warp
                continue
            
            print(f"   [Scale] Factor: {calculated_scale:.4f}")
            scale = calculated_scale

        # Apply Scale
        if M2_opt is not None: M2_opt[:3, 3] *= scale
        if P_warp is not None: P_warp *= scale
        
        # 7. Accumulate Global Pose
        N_pts = P_warp.shape[0]
        P_homo = np.hstack((P_warp, np.ones((N_pts, 1))))
        P_world = (T_global @ P_homo.T).T
        P_world = P_world[:, :3]

        reconstruction_data.append({
            'id': i,
            'P_world': P_world,
            'img': img1, 
            'pts': pts1_in_warp
        })

        # Update Transformation
        T_rel = np.eye(4)
        T_rel[:3, :] = M2_opt 
        T_global = T_global @ np.linalg.inv(T_rel)
        
        prev_img_pts = pts2_in_warp  
        prev_P_dense = P_warp

    # --- Export ---
    if len(reconstruction_data) > 0:
        out_file = "reconstruction_results_fullres.npz"
        print(f"\nExporting to {out_file}...")
        # Save explicitly as object array to avoid pickle issues
        np.savez_compressed(out_file, data=np.array(reconstruction_data, dtype=object))
        print("Export complete.")

if __name__ == "__main__":
    run_image_pipeline()


if __name__ == "__main__":
    run_correctness_test()
    run_benchmark()
    run_image_pipeline()