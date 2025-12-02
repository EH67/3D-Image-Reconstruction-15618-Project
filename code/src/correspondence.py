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

# Import any modules defined by PYBIND11_MODULE from src/bindings.cpp.
import cuda_ops_module 
import cuda_ransac_module

NUM_ITERS = 5000
THRESHOLD = 3.0

def get_correspondences_cpu(img1, img2, M, num_iters=NUM_ITERS, threshold=THRESHOLD):
    #feature matching
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    #knn matching using FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1_candidates = []
    pts2_candidates = []

    #ratio testing: ensuring best point is significantly different from second best
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

            pts1_candidates.append(kp1[m.queryIdx].pt)
            pts2_candidates.append(kp2[m.trainIdx].pt)

    pts1_candidates = np.float32(pts1_candidates)
    pts2_candidates = np.float32(pts2_candidates)

    print(f"Found {len(pts1_candidates)} raw matches after Ratio Test.")
    if len(pts1_candidates) < 8:
        print("Not enough matches to compute fundamental matrix Using 8 point algorithm.")
        return None, None, None

  #ransac
    F, mask = cv2.findFundamentalMat(pts1_candidates, pts2_candidates, cv2.FM_RANSAC, threshold, 0.99)
    F_2, mask_2 = ransac_fundamental_matrix(pts1_candidates, pts2_candidates, M, num_iters, threshold)
    print("F", F.shape)
    print(F)
    print("F_2", F_2.shape)
    print(F_2)


    #only use inliers
    pts1 = pts1_candidates[mask_2 == 1]
    pts2 = pts2_candidates[mask_2 == 1]
    print("pts1.shape", pts1.shape, "pts2.shape", pts2.shape, "mask_2.shape", mask_2.shape)

    inlier_matches_vis_new = []
    for i, match in enumerate(good_matches):
        if mask_2[i]:
            inlier_matches_vis_new.append(match)

    # visualize_inliers(im1, kp1, im2, kp2, inlier_matches_vis)

    #   return pts1, pts2, F
    return pts1, pts2, F_2



def get_correspondences_gpu(img1, img2, M, num_iters=NUM_ITERS, threshold=THRESHOLD):
    #feature matching (serial)
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    #knn matching using FLANN
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1_candidates = []
    pts2_candidates = []

    #ratio testing: ensuring best point is significantly different from second best
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

            pts1_candidates.append(kp1[m.queryIdx].pt)
            pts2_candidates.append(kp2[m.trainIdx].pt)

    pts1_candidates = np.float32(pts1_candidates)
    pts2_candidates = np.float32(pts2_candidates)
    N = pts1_candidates.shape[0]

    print(f"Found {len(pts1_candidates)} raw matches after Ratio Test.")
    if len(pts1_candidates) < 8:
        print("Not enough matches to compute fundamental matrix Using 8 point algorithm.")
        return None, None, None

    #ransac (parallel).
    F = cuda_ransac_module.cuda_ransac(pts1_candidates, pts2_candidates, int(M), num_iters, threshold)
    print("cuda F", F)

    # Get inlier mask (sequential). #TODO make this parallel
    hpts1 = np.concatenate([pts1_candidates, np.ones([N, 1])], axis=1) # Nx3
    hpts2 = np.concatenate([pts2_candidates, np.ones([N, 1])], axis=1) # Nx3
    errors = compute_symmetric_epipolar_distance(F, hpts1, hpts2)
    mask = errors < THRESHOLD ** 2


    #only use inliers. # TODO combine with above.
    pts1 = pts1_candidates[mask == 1]
    pts2 = pts2_candidates[mask == 1]
    print("pts1.shape", pts1.shape, "pts2.shape", pts2.shape, "mask_2.shape", mask.shape)

    inlier_matches = []
    for i, match in enumerate(good_matches):
        if mask[i]:
            inlier_matches.append(match)

    # visualize_inliers(im1, kp1, im2, kp2, inlier_matches)

    #   return pts1, pts2, F
    return pts1, pts2, F
