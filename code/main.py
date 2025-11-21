import sys
import os
import numpy as np
import time
import cv2

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.

# Import any modules defined by PYBIND11_MODULE from src/bindings.cpp.
import cuda_ops_module 

#newer version of finding epipolar correspondences that uses SIFT and FLAAN to be faster
def get_correspondences(img1, img2):
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
  F, mask = cv2.findFundamentalMat(pts1_candidates, pts2_candidates, cv2.FM_RANSAC, 3.0, 0.99)

  #only use inliers
  pts1 = pts1_candidates[mask.ravel() == 1]
  pts2 = pts2_candidates[mask.ravel() == 1]

  inlier_matches_vis = []
  mask_flat = mask.ravel()
  for i, match in enumerate(good_matches):
      if mask_flat[i]:
          inlier_matches_vis.append(match)

  # visualize_inliers(im1, kp1, im2, kp2, inlier_matches_vis)

  return pts1, pts2, F




IMAGE_DIR = os.path.abspath(os.path.join(current_dir, 'images'))
IM1_PATH = os.path.join(IMAGE_DIR, 'bag3.jpg')
IM2_PATH = os.path.join(IMAGE_DIR, 'bag4.jpg')

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

#find correspondences
pts1_in, pts2_in, F_auto = get_correspondences(im1, im2)