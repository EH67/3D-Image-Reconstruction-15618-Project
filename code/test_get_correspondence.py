"""
Not used often - tests the correspondence pipeline to ensure correctness.

This was used in the process of implementing the pipeline to ensure intermediate steps
are correct. But this does not yield any useful metrics.
"""

import sys
import os
import numpy as np
import time
import cv2
import scipy

from src.ransac import ransac_fundamental_matrix, _singularize, compute_symmetric_epipolar_distance
from src.correspondence import get_correspondences_cpu, get_correspondences_gpu

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.

# Import any modules defined by PYBIND11_MODULE from src/bindings.cpp.
import cuda_ops_module 
import cuda_ransac_module


EXPORT_OUTPUT = False # Creates .npz for the correspondance output to be plotted using Colab


def export_visualization_data(filename, img1, img2, pts1, pts2):
    """
    Saves images and inlier coordinates to a .npz file.
    """
    print(f"Exporting data to {filename}...")
    # Ensure points are numpy arrays
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    # Save compressed to save space
    np.savez_compressed(
        filename,
        img1=img1,
        img2=img2,
        pts1=pts1,
        pts2=pts2
    )
    print("Export complete.")


def compare_matrices(F1, F2):
    """
    Compares two Fundamental matrices for similarity.
    Since F is scale-invariant, we normalize them before comparing.
    Also handles the sign ambiguity (F is equivalent to -F).
    """
    if F1 is None or F2 is None:
        return float('inf')

    # 1. Normalize both matrices to have Frobenius norm = 1
    F1_norm = F1 / np.linalg.norm(F1)
    F2_norm = F2 / np.linalg.norm(F2)

    # 2. Handle sign ambiguity: If the directions are opposite, flip one
    # We check the sign of the first non-zero element or simply dot product
    if np.sum(F1_norm * F2_norm) < 0:
        F2_norm = -F2_norm

    # 3. Compute Frobenius distance (element-wise Euclidean distance)
    diff = np.linalg.norm(F1_norm - F2_norm)
    return diff


if __name__=='__main__':
    IMAGE_DIR = os.path.abspath(os.path.join(current_dir, 'images'))
    IM1_PATH = os.path.join(IMAGE_DIR, 'bag3.jpg')
    IM2_PATH = os.path.join(IMAGE_DIR, 'bag4.jpg')
    print(IM1_PATH)

    # Load images as BGR for OpenCV functions, then convert to RGB for Matplotlib
    im1 = cv2.imread(IM1_PATH)
    im2 = cv2.imread(IM2_PATH)
    if im1 is None or im2 is None:
        raise FileNotFoundError(f"Could not find {IM1_PATH} or {IM2_PATH}. Please check file paths.")

    im1_rgb = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2_rgb = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    h, w, _ = im1.shape
    M = np.max([w, h])

    # Approximate camera intrinsics of the media sent in.
    focal_length = w * 1.2
    cx = w / 2.0
    cy = h / 2.0

    K_mock = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])


    intrinsics_mock = {'K1': K_mock, 'K2': K_mock}
    K1, K2 = K_mock, K_mock

    print(f"Image Resolution: {w}x{h}")
    print("Mock Intrinsics K:\n", K1.round(2))

    # --- 1. Run CPU Benchmark ---
    print("Running CPU Implementation...")
    start_cpu = time.perf_counter()
    
    pts1_in, pts2_in, F_cpu = get_correspondences_cpu(im1, im2, M)
    
    end_cpu = time.perf_counter()
    time_cpu = end_cpu - start_cpu
    
    num_inliers_cpu = len(pts1_in) if pts1_in is not None else 0
    print(f"CPU Finished. Time: {time_cpu:.4f}s | Inliers: {num_inliers_cpu}")
    print("-" * 50)

    # --- 2. Run GPU Benchmark ---
    print("Running GPU Implementation...")
    # Warmup
    get_correspondences_gpu(im1, im2, M)
    
    start_gpu = time.perf_counter()
    
    pts1_in_gpu, pts2_in_gpu, F_gpu = get_correspondences_gpu(im1, im2, M)
    
    end_gpu = time.perf_counter()
    time_gpu = end_gpu - start_gpu
    
    num_inliers_gpu = len(pts1_in_gpu) if pts1_in_gpu is not None else 0
    print(f"GPU Finished. Time: {time_gpu:.4f}s | Inliers: {num_inliers_gpu}")
    print("-" * 50)

    # --- 3. Analysis & Export ---
    
    # Speedup Calculation
    if time_gpu > 0:
        speedup = time_cpu / time_gpu
        print(f"Speedup: {speedup:.2f}x")
    
    # Correctness Check
    # Note: RANSAC is randomized, so F matrices will rarely be identical.
    # We check if they are geometrically close.
    diff = compare_matrices(F_cpu, F_gpu)
    print(f"Matrix Similarity Score (Frobenius Norm Diff): {diff:.4f}")
    
    if diff < 0.1:
        print(">> SUCCESS: Matrices are very similar.")
    elif diff < 0.5:
        print(">> ACCEPTABLE: Matrices are somewhat similar (expected due to RANSAC randomness).")
    else:
        print(">> WARNING: Matrices diverged significantly.")

    # Inlier Count check
    if num_inliers_cpu > 0:
        inlier_ratio = num_inliers_gpu / num_inliers_cpu
        print(f"Inlier Count Ratio (GPU/CPU): {inlier_ratio:.2f}")

    # Export output if applicable (upload to Colab to plot, need to have the function "plot_saved_results").
    if EXPORT_OUTPUT:
        if pts1_in is not None:
            export_visualization_data("results_cpu.npz", im1, im2, pts1_in, pts2_in)

        if pts1_in_gpu is not None:
            export_visualization_data("results_gpu.npz", im1, im2, pts1_in_gpu, pts2_in_gpu)