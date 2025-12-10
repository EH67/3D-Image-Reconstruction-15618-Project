# Completely Serial RANSAC Algorithm.

import numpy as np
import scipy
import time

MIN_SAMPLE_SIZE = 8

def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
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
        F_model = eightpoint_minimal(sample_pts1, sample_pts2, M)
        
        # 3. Error Calculation (Symmetric Epipolar Distance)
        # Calculate distance for ALL N points against F_model
        errors = compute_symmetric_epipolar_distance(F_model, hpts1, hpts2)

        # 4. Consensus Check
        # Find inliers: points whose error is less than the threshold
        current_mask = errors < (threshold ** 2)
        current_inliers_count = np.sum(current_mask)

        # 5. Model Update
        if current_inliers_count > max_inliers:
            max_inliers = current_inliers_count
            best_mask = current_mask
            best_F = F_model

    return best_F, best_mask.astype(np.uint8)


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
    start = time.time()
    # F * p1 (3x3 * 3xN = 3xN) -> Epipolar lines l2 in image 2
    l2 = F @ hpts1.T
    
    # F.T * p2 (3x3 * 3xN = 3xN) -> Epipolar lines l1 in image 1
    l1 = F.T @ hpts2.T
    
    # Epipolar Constraint: p2.T * F * p1 (1xN) -> distance from point to line (numerator of distance formula)
    numerator = np.sum(hpts2.T * l2, axis=0)
    
    # Denominators: squared norm of the first two components of the line vector
    denom2 = l2[0, :]**2 + l2[1, :]**2
    denom1 = l1[0, :]**2 + l1[1, :]**2
    
    # Symmetric Epipolar Distance (SED) formula
    errors = (numerator**2 / denom2) + (numerator**2 / denom1)
    
    return errors