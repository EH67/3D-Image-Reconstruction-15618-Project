"""
Test pipeline for the correspondence workflow.

This pipeline gives times for each of the preprocessing steps, and compares the correspondence
pipeline (ransac + triangulation) of the CPU numpy vs GPU CUDA implementation. 
"""

import sys
import os
import numpy as np
import time
import cv2

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, current_dir)

# --- Module Imports ---
# Updated imports to use separated functions
from src.correspondence import extract_features, compute_matches, filter_matches, run_ransac_cpu, run_ransac_gpu, run_ransac_warp_gpu

try:
    from src.find_M2 import findM2_cpu, findM2_gpu
except ImportError:
    from find_m2 import findM2_cpu, findM2_gpu

EXPORT_OUTPUT = False  # Set to True to save .npz files for Colab plotting
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

if __name__ == '__main__':
    cv2.setRNGSeed(0)
    np.random.seed(0)

    # --- 1. Load Data ---
    IMAGE_DIR = os.path.abspath(os.path.join(current_dir, 'images/Pokemon_Plush'))
    IM1_PATH = os.path.join(IMAGE_DIR, 'PokePlush_09.jpg')
    IM2_PATH = os.path.join(IMAGE_DIR, 'PokePlush_10.jpg')

    im1 = cv2.imread(IM1_PATH)
    im2 = cv2.imread(IM2_PATH)
    if RESIZE:
        im1 = resize_with_aspect_ratio(im1, width=1200)
        im2 = resize_with_aspect_ratio(im2, width=1200)

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

    # --- 2. Common Pre-processing (SIFT + Matching) ---
    print("[COMMON] Pre-processing Steps...")
    
    # Step A: Feature Extraction
    t0 = time.perf_counter()
    kp1, des1 = extract_features(im1)
    kp2, des2 = extract_features(im2)
    t1 = time.perf_counter()
    
    # Step B: KNN Matching
    matches_raw = compute_matches(des1, des2)
    t2 = time.perf_counter()
    
    # Step C: Ratio Test Filtering
    pts1_cand, pts2_cand = filter_matches(matches_raw, kp1, kp2)
    t3 = time.perf_counter()
    
    time_extract = t1 - t0
    time_knn = t2 - t1
    time_filter = t3 - t2
    time_common_total = t3 - t0
    
    print(f"    - SIFT Extraction Time: {time_extract:.4f}s")
    print(f"    - KNN Matching Time:    {time_knn:.4f}s")
    print(f"    - Ratio Test Time:      {time_filter:.4f}s")
    print(f"    - Total Pre-processing: {time_common_total:.4f}s")
    print(f"    - Matches Found:        {len(pts1_cand)}")
    print("-" * 60)

    if len(pts1_cand) < 8:
        print("Error: Not enough matches to proceed.")
        sys.exit(0)

    # --- 3. CPU Pipeline (RANSAC + M2) ---
    print("[CPU] Starting Optimization (RANSAC + FindM2)...")
    start_cpu = time.perf_counter()
    
    # A. RANSAC
    F_cpu, mask_cpu = run_ransac_cpu(pts1_cand.copy(), pts2_cand.copy(), M_scale, num_iters=5000, threshold=3.0)
    
    # Apply Mask
    mask_cpu = mask_cpu.ravel().astype(bool)
    pts1_cpu_inliers = pts1_cand[mask_cpu]
    pts2_cpu_inliers = pts2_cand[mask_cpu]
    
    t_ransac_cpu = time.perf_counter()

    # B. Find M2 & Triangulate
    if F_cpu is not None:
        M2_cpu, C2_cpu, P_cpu = findM2_cpu(F_cpu, pts1_cpu_inliers, pts2_cpu_inliers, intrinsics)
    else:
        M2_cpu, P_cpu = None, None

    end_cpu = time.perf_counter()
    
    time_cpu_ransac = t_ransac_cpu - start_cpu
    time_cpu_m2 = end_cpu - t_ransac_cpu
    time_cpu_total_ops = end_cpu - start_cpu

    print(f"    - RANSAC Time: {time_cpu_ransac:.4f}s")
    print(f"    - FindM2 Time: {time_cpu_m2:.4f}s")
    print(f"    - Total Optimization Time: {time_cpu_total_ops:.4f}s")
    print("-" * 60)

    # --- 4. GPU Pipeline (RANSAC Block Version + M2) ---
    # Warmup
    run_ransac_gpu(pts1_cand[:8], pts2_cand[:8], M_scale, num_iters=5, threshold=3.0)
    print("[GPU] Starting Optimization (RANSAC Block Version + FindM2)...")
    start_gpu = time.perf_counter()
    
    # A. RANSAC
    F_gpu, mask_gpu = run_ransac_gpu(pts1_cand, pts2_cand, M_scale, num_iters=5000, threshold=3.0)
    
    # Apply Mask
    mask_gpu = mask_gpu.ravel().astype(bool)
    pts1_gpu_inliers = pts1_cand[mask_gpu]
    pts2_gpu_inliers = pts2_cand[mask_gpu]
    
    t_ransac_gpu = time.perf_counter()

    # B. Find M2 & Triangulate
    if F_gpu is not None:
        M2_gpu, C2_gpu, P_gpu = findM2_gpu(F_gpu, pts1_gpu_inliers, pts2_gpu_inliers, intrinsics)
    else:
        M2_gpu, P_gpu = None, None

    end_gpu = time.perf_counter()
    
    time_gpu_ransac = t_ransac_gpu - start_gpu
    time_gpu_m2 = end_gpu - t_ransac_gpu
    time_gpu_total_ops = end_gpu - start_gpu

    print(f"    - RANSAC Time: {time_gpu_ransac:.4f}s")
    print(f"    - FindM2 Time: {time_gpu_m2:.4f}s")
    print(f"    - Total Optimization Time: {time_gpu_total_ops:.4f}s")
    print("-" * 60)

    # --- 5. GPU Pipeline (RANSAC Warp Version + M2) ---
    # Warmup
    run_ransac_warp_gpu(pts1_cand[:8], pts2_cand[:8], M_scale, num_iters=5, threshold=3.0)
    print("[GPU] Starting Optimization (RANSAC Warp Version + FindM2)...")
    start_warp_gpu = time.perf_counter()
    
    # A. RANSAC
    F_warp_gpu, mask_warp_gpu = run_ransac_warp_gpu(pts1_cand, pts2_cand, M_scale, num_iters=5000, threshold=3.0)
    
    # Apply Mask
    mask_warp_gpu = mask_warp_gpu.ravel().astype(bool)
    pts1_warp_gpu_inliers = pts1_cand[mask_warp_gpu]
    pts2_warp_gpu_inliers = pts2_cand[mask_warp_gpu]
    
    t_ransac_warp_gpu = time.perf_counter()

    # B. Find M2 & Triangulate
    if F_warp_gpu is not None:
        M2_warp_gpu, C2_warp_gpu, P_warp_gpu = findM2_gpu(F_warp_gpu, pts1_warp_gpu_inliers, pts2_warp_gpu_inliers, intrinsics)
    else:
        M2_warp_gpu, P_warp_gpu = None, None

    end_warp_gpu = time.perf_counter()
    
    time_warp_gpu_ransac = t_ransac_warp_gpu - start_warp_gpu
    time_warp_gpu_m2 = end_warp_gpu - t_ransac_warp_gpu
    time_warp_gpu_total_ops = end_warp_gpu - start_warp_gpu

    print(f"    - RANSAC Warp Time: {time_warp_gpu_ransac:.4f}s")
    print(f"    - FindM2 Time: {time_warp_gpu_m2:.4f}s")
    print(f"    - Total Optimization Time: {time_warp_gpu_total_ops:.4f}s")
    print("-" * 60)

    # --- 5. Analysis ---
    print("SUMMARY RESULTS (Optimization Only)")
    
    # Speedup Calculation (Excluding SIFT/Matching)
    if time_gpu_total_ops > 0:
        speedup = time_cpu_total_ops / time_gpu_total_ops
        print(f"Optimization Speedup: {speedup:.2f}x")
    else:
        print("Optimization Speedup: N/A (GPU Time is 0)")

    # Correctness Checks
    compare_matrices_frobenius(F_cpu, F_gpu, "Fundamental Matrix (F)")
    compare_matrices_frobenius(M2_cpu, M2_gpu, "Camera Matrix (M2)")

    if P_cpu is not None and P_gpu is not None:
        print(f">> CPU Points: {len(P_cpu)}")
        print(f">> GPU Points: {len(P_gpu)}")
        # Check simple centroid distance
        dist = np.linalg.norm(np.mean(P_cpu, axis=0) - np.mean(P_gpu, axis=0))
        print(f">> Point Cloud Centroid Distance: {dist:.4f}")


    if EXPORT_OUTPUT:
        print("-" * 60)
        print("EXPORTING RESULTS")

        # Convert to RGB for export/visualization (OpenCV loads as BGR)
        im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        
        if P_cpu is not None:
            export_visualization_data("results_cpu.npz", im1_rgb, im2_rgb, pts1_cpu_inliers, pts2_cpu_inliers, P_cpu)
        
        if P_gpu is not None:
            export_visualization_data("results_gpu_warp.npz", im1_rgb, im2_rgb, pts1_gpu_inliers, pts2_gpu_inliers, P_gpu)