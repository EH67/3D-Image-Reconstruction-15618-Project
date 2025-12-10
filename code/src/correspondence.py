import sys
import os
import numpy as np
import time
import cv2
import scipy

from src.ransac import ransac_fundamental_matrix, _singularize, compute_symmetric_epipolar_distance

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
build_dir = os.path.join(parent_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.


import cuda_ops_module 
import cuda_ransac_module
import cuda_ransac_warp_module

NUM_ITERS = 5000
THRESHOLD = 3.0
def extract_features(img):
    """
    Detects features and computes descriptors using SIFT.
    Returns: kp, des
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def compute_matches(des1, des2):
    """
    Computes KNN matches using FLANN.
    Returns: matches (list of DMatch objects)
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return matches

def filter_matches(matches, kp1, kp2, ratio=0.8):
    """
    Filters matches using the ratio test.
    Returns: pts1, pts2 (Nx2 numpy arrays of matching points)
    """
    pts1_candidates = []
    pts2_candidates = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            pts1_candidates.append(kp1[m.queryIdx].pt)
            pts2_candidates.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1_candidates)
    pts2 = np.float32(pts2_candidates)
    
    return pts1, pts2

def run_ransac_cpu(pts1, pts2, M, num_iters, threshold):
    """
    Executes RANSAC using the CPU Python implementation.
    Returns: F, mask (boolean array)
    """
    # print("RANSAC CPU called for num_iters", num_iters, "threshold", threshold)
    F, mask = ransac_fundamental_matrix(pts1, pts2, M, num_iters, threshold)
    mask = mask.astype(bool)
    # print(F)
    return F, mask

def run_ransac_gpu(pts1, pts2, M, num_iters, threshold):
    """
    Executes RANSAC using the CUDA implementation.
    Returns: F, mask (boolean array)
    """
    F = cuda_ransac_module.cuda_ransac(pts1, pts2, int(M), num_iters, threshold)
    
    # Compute Mask (Sequential calculation of error)
    N = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([N, 1])], axis=1) # Nx3
    hpts2 = np.concatenate([pts2, np.ones([N, 1])], axis=1) # Nx3
    
    errors = compute_symmetric_epipolar_distance(F, hpts1, hpts2)
    mask = errors < threshold ** 2
    
    return F, mask

def run_ransac_warp_gpu(pts1, pts2, M, num_iters, threshold):
    """
    Executes RANSAC using the CUDA Warp implementation.
    Returns: F, mask (boolean array)
    """
    F = cuda_ransac_warp_module.cuda_ransac_warp(pts1, pts2, int(M), num_iters, threshold)
    
    # Compute Mask (Sequential calculation of error)
    N = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([N, 1])], axis=1) # Nx3
    hpts2 = np.concatenate([pts2, np.ones([N, 1])], axis=1) # Nx3
    
    errors = compute_symmetric_epipolar_distance(F, hpts1, hpts2)
    mask = errors < threshold ** 2
    
    return F, mask

def _process_correspondences(img1, img2, M, ransac_func, num_iters, threshold):
    """
    Generic pipeline that accepts a specific RANSAC function.
    """
    # 1. Feature Extraction
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)

    # 2. Matching (Split steps)
    matches = compute_matches(des1, des2)
    pts1_cand, pts2_cand = filter_matches(matches, kp1, kp2)


    if len(pts1_cand) < 8:
        print("Not enough matches to compute fundamental matrix (Need >= 8).")
        return None, None, None

    # 3. RANSAC
    F, mask = ransac_func(pts1_cand, pts2_cand, M, num_iters, threshold)

    if F is None:
        print("RANSAC failed to find a Fundamental Matrix.")
        return None, None, None

    # 4. Filter Inliers
    mask = mask.ravel() 
    pts1_inliers = pts1_cand[mask]
    pts2_inliers = pts2_cand[mask]

    # print(f"Inliers: {len(pts1_inliers)} / {len(pts1_cand)}")
    
    return pts1_inliers, pts2_inliers, F

# --- Main Callable Functions ---

def get_correspondences_cpu(img1, img2, M, num_iters=NUM_ITERS, threshold=THRESHOLD):
    return _process_correspondences(img1, img2, M, run_ransac_cpu, num_iters, threshold)

def get_correspondences_gpu(img1, img2, M, num_iters=NUM_ITERS, threshold=THRESHOLD):
    return _process_correspondences(img1, img2, M, run_ransac_gpu, num_iters, threshold)