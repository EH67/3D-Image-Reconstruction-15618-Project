import sys
import os
import numpy as np
import time
import cv2
import scipy

# Below code are needed in order to load the cuda modules we wrote/compiled (stored in /build).
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, 'build') # add /build to current abs path.
sys.path.insert(0, build_dir) # add path with /build to sys path.

# Import any modules defined by PYBIND11_MODULE from src/bindings.cpp.
import cuda_ops_module 

def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F

def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
        disp=False
    )
    return _singularize(f.reshape([3, 3]))

def eightpoint(pts1, pts2, M):
  '''
  Q2.1: Eight Point Algorithm
  Input:  pts1, Nx2 Matrix
          pts2, Nx2 Matrix
          M, a scalar parameter computed as max(imwidth, imheight)
  Output: F, the fundamental matrix

  HINTS:
  (1) Normalize the input pts1 and pts2 using the matrix T.
  (2) Setup the eight point algorithm's equation.
  (3) Solve for the least square solution using SVD.
  (4) Use the function `_singularize` (provided in the helper functions above) to enforce the singularity condition.
  (5) Use the function `refineF` (provided in the helper functions above) to refine the computed fundamental matrix.
      (Remember to use the normalized points instead of the original points)
  (6) Unscale the fundamental matrix by the lower right corner element
  '''


  F = None
  N = pts1.shape[0]

  T = np.array([[1.0/M, 0,     0],
                [0,     1.0/M, 0],
                [0,     0,     1]])

  #normalize
  pts1_norm = pts1 / M
  pts2_norm = pts2 / M

  #setup
  x1 = pts1_norm[:, 0]
  y1 = pts1_norm[:, 1]
  x2 = pts2_norm[:, 0]
  y2 = pts2_norm[:, 1]

  #make A
  A = np.stack([
      x2 * x1,
      x2 * y1,
      x2,
      y2 * x1,
      y2 * y1,
      y2,
      x1,
      y1,
      np.ones(N)
  ], axis=1)


  U, S, Vh = np.linalg.svd(A)
  f = Vh[-1, :]
  F_norm = f.reshape(3, 3)

  #singularize
  F_norm = _singularize(F_norm)

  #refinef
  F_norm = refineF(F_norm, pts1_norm, pts2_norm)

  #unscale by lower corner elem
  F = T.T @ F_norm @ T
  if F[2, 2] != 0:
      F = F / F[2, 2]
  # print(F[2,2])


  return F

def ransac_fundamental_matrix(pts1, pts2, M, num_iterations=1000, threshold=3.0):
    '''
    Performs RANSAC to find the Fundamental Matrix (F) and the set of inliers.
    This simulates the behavior of cv2.findFundamentalMat with FM_RANSAC.
    '''
    N = pts1.shape[0]
    best_F = np.zeros((3, 3))
    max_inliers = -1
    best_mask = np.zeros(N, dtype=bool)

    # Pre-allocate hpts to avoid repeated concatenation inside the loop
    hpts1 = np.concatenate([pts1, np.ones([N, 1])], axis=1) # Nx3
    hpts2 = np.concatenate([pts2, np.ones([N, 1])], axis=1) # Nx3

    for _ in range(num_iterations):
        # 1. Random Sample (S=8 points)
        # Select 8 random indices
        indices = np.random.choice(N, MIN_SAMPLE_SIZE, replace=False)
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]

        # 2. Model Estimation (Eight-Point Algorithm on the minimal set)
        # Note: We must call a stripped-down 8-point algorithm here that doesn't
        # use the non-linear refinement, as it's too slow for the RANSAC loop.
        F_model = eightpoint_minimal(sample_pts1, sample_pts2, M)
        
        # 3. Error Calculation (Symmetric Epipolar Distance)
        # Calculate distance for ALL N points against F_model
        errors = compute_symmetric_epipolar_distance(F_model, hpts1, hpts2)

        # 4. Consensus Check
        # Find inliers: points whose error is less than the threshold
        current_mask = errors < threshold
        current_inliers_count = np.sum(current_mask)

        # 5. Model Update
        if current_inliers_count > max_inliers:
            max_inliers = current_inliers_count
            best_mask = current_mask
            best_F = F_model

    # 6. Final Refinement (Optional but recommended, using all inliers)
    # Use only the best inliers to run the full eightpoint algorithm once
    if max_inliers > MIN_SAMPLE_SIZE:
        final_F = eightpoint(pts1[best_mask], pts2[best_mask], M)
    else:
        final_F = best_F
        
    return final_F, best_mask.astype(np.uint8)


def eightpoint_minimal(pts1, pts2, M):
    '''
    Stripped-down version of eightpoint for fast RANSAC loop execution.
    Only computes LSE, singularizes, and unscales. Skips refinement.
    '''
    N = pts1.shape[0]
    T = np.array([[1.0/M, 0,     0],
                  [0,     1.0/M, 0],
                  [0,     0,     1]])

    pts1_norm = pts1 / M
    pts2_norm = pts2 / M

    x1, y1 = pts1_norm[:, 0], pts1_norm[:, 1]
    x2, y2 = pts2_norm[:, 0], pts2_norm[:, 1]

    A = np.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, np.ones(N)
    ], axis=1)

    _, _, Vh = np.linalg.svd(A)
    f = Vh[-1, :]
    F_norm = f.reshape(3, 3)
    
    F_norm_rank2 = _singularize(F_norm)

    F_unscaled = T.T @ F_norm_rank2 @ T
    if F_unscaled[2, 2] != 0:
        F_unscaled = F_unscaled / F_unscaled[2, 2]
    
    return F_unscaled
    

def compute_symmetric_epipolar_distance(F, hpts1, hpts2):
    ''' 
    Computes the symmetric epipolar distance for all N points using a vectorized approach.
    This is the core error metric used by RANSAC.
    '''
    # F * p1 (3x3 * 3xN = 3xN) -> Epipolar lines l2 in image 2
    l2 = F @ hpts1.T
    
    # F.T * p2 (3x3 * 3xN = 3xN) -> Epipolar lines l1 in image 1
    l1 = F.T @ hpts2.T
    
    # Epipolar Constraint: p2.T * F * p1 (1xN) -> distance from point to line (numerator of distance formula)
    # np.sum(hpts2.T * l2, axis=0) is equivalent to the matrix multiplication trace(hpts2 @ F @ hpts1.T)
    numerator = np.sum(hpts2.T * l2, axis=0)
    
    # Denominators: squared norm of the first two components of the line vector
    # l2[0, :]**2 + l2[1, :]**2 -> Squared length of line vector components for image 2
    # l1[0, :]**2 + l1[1, :]**2 -> Squared length of line vector components for image 1
    
    denom2 = l2[0, :]**2 + l2[1, :]**2
    denom1 = l1[0, :]**2 + l1[1, :]**2
    
    # Symmetric Epipolar Distance (SED) formula
    # SED = ( (p2.T * F * p1)^2 / (l2_x^2 + l2_y^2) ) + ( (p1.T * F.T * p2)^2 / (l1_x^2 + l1_y^2) )
    errors = (numerator**2 / denom2) + (numerator**2 / denom1)
    
    # The error we compare against the threshold is typically the square root (pixels),
    # but since the threshold is also squared, we can return the squared error.
    return errors

#newer version of finding epipolar correspondences that uses SIFT and FLAAN to be faster.
def get_correspondences(img1, img2, M):
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
  F_2 = eightpoint(pts1_candidates, pts2_candidates, M)
  print("F", F.shape)
  print(F)
  print("F_2", F_2.shape)
  print(F_2)


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

  #find correspondences
  pts1_in, pts2_in, F_auto = get_correspondences(im1, im2, M)