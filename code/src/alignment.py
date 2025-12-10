"""
Module for all the postprocessing (bundle adjustment, scale adjustment).
"""

import math
from scipy.spatial import KDTree
import scipy.optimize
from scipy.optimize import least_squares
import numpy as np

def rodrigues(r):
    r = np.array(r).flatten()
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        U = (r/theta)[:, np.newaxis]
        Ux, Uy, Uz = r/theta
        K = np.array([[0, -Uz, Uy], [Uz, 0, -Ux], [-Uy, Ux, 0]])
        R = np.eye(3) * np.cos(theta) + np.sin(theta) * K + \
            (1 - np.cos(theta)) * np.matmul(U, U.T)
    return R

def invRodrigues(R):
    A = (R - R.T)/2
    ro = [A[2, 1], A[0, 2], A[1, 0]]
    s = np.linalg.norm(ro)
    c = (np.trace(R) - 1)/2
    if s == 0 and c == 1:
        r = np.zeros(3)
    elif s == 0 and c == -1:
        col = np.eye(3) + R
        # Find first non-zero column
        col_idx = np.where(np.sum(np.abs(col), axis=0) > 0)[0][0]
        v = col[:, col_idx]
        u = v/np.linalg.norm(v)
        r = u * np.pi
        # Enforce semi-circle constraint if needed
        if not ((r[0]>0) or (r[0]==0 and r[1]>0) or (r[0]==0 and r[1]==0 and r[2]>0)):
             r = -r
    else:
        u = ro/s
        theta = np.arctan2(s, c)
        r = u * theta
    return r

def rodriguesResidual(K1, M1, p1, K2, p2, x):
  if K1 is None or K2 is None or M1 is None or p1 is None or p2 is None:
    # If any essential input is None, log the error and return NaN residuals
    # This prevents the optimizer from crashing but flags a major setup issue.
    print(f"BA Residual Error: K1={K1 is None}, K2={K2 is None}, M1={M1 is None}")
    # Return high residuals (a large array of NaNs) to penalize this step
    N_total_residuals = p1.shape[0] * 4 # 2 points per 2 cameras
    return np.full(N_total_residuals, np.nan)

  N = p1.shape[0]
  P = x[0 : 3*N].reshape(N, 3)
  r2 = x[3*N : 3*N + 3]
  t2 = x[3*N + 3 : 3*N + 6]

  # Project into Camera 1
  P_homo = np.hstack([P, np.ones((N, 1))])
  C1 = K1 @ M1
  p1_proj_homo = (C1 @ P_homo.T).T
  p1_proj = p1_proj_homo[:, 0:2] / p1_proj_homo[:, 2, np.newaxis]

  # Project into Camera 2
  R2 = rodrigues(r2)
  M2 = np.hstack([R2, t2.reshape(3, 1)])
  C2 = K2 @ M2
  p2_proj_homo = (C2 @ P_homo.T).T
  p2_proj = p2_proj_homo[:, 0:2] / p2_proj_homo[:, 2, np.newaxis]

  # Compute residuals
  residuals_p1 = (p1 - p1_proj).ravel()
  residuals_p2 = (p2 - p2_proj).ravel()
  
  return np.concatenate((residuals_p1, residuals_p2))

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    N = P_init.shape[0]
    
    # Extract initial rotation and translation from M2_init
    R2_init = M2_init[:, 0:3]
    t2_init = M2_init[:, 3]
    r2_init = invRodrigues(R2_init)
    
    # Construct initial parameter vector (Points, rotation, translation)
    x0 = np.concatenate((P_init.flatten(), r2_init.flatten(), t2_init.flatten()))

    def fun(x):
        return rodriguesResidual(K1, M1, p1, K2, p2, x)

    # Run optimization
    res = least_squares(fun, x0, method='trf', x_scale='jac', verbose=2, ftol=1e-4, loss='soft_l1')
    
    x_opt = res.x
    
    # Unpack results
    P_opt = x_opt[0 : 3*N].reshape(N, 3)
    r2_opt = x_opt[3*N : 3*N + 3]
    t2_opt = x_opt[3*N + 3 : 3*N + 6]
    
    R2_opt = rodrigues(r2_opt)
    M2_opt = np.hstack([R2_opt, t2_opt.reshape(3, 1)])
    
    return M2_opt, P_opt


#for consecutive frames 1,2,3, they share frame 2.
#using these shared points, we scale the points so they are on 
#the same world frame
def get_scale_factor_strict(prev_pts_in_overlap, prev_P_3d, curr_pts_in_overlap, curr_P_3d):
   
    if len(prev_pts_in_overlap) < 10 or len(curr_pts_in_overlap) < 10:
        return None

    tree = KDTree(prev_pts_in_overlap)
    distances, indices = tree.query(curr_pts_in_overlap, distance_upper_bound=2.0)
    
    valid_match = distances != float('inf')
    if np.sum(valid_match) < 10: # Require at least 10 common points
        return None

    idx_prev = indices[valid_match]
    idx_curr = np.where(valid_match)[0]
    
    p3d_prev = prev_P_3d[idx_prev]
    p3d_curr = curr_P_3d[idx_curr]
    
    # Compute pairwise distances 
    if len(p3d_prev) > 500:
        p3d_prev = p3d_prev[:500]
        p3d_curr = p3d_curr[:500]

    dists_prev = np.linalg.norm(p3d_prev[:, None] - p3d_prev, axis=2).flatten()
    dists_curr = np.linalg.norm(p3d_curr[:, None] - p3d_curr, axis=2).flatten()
    
    nonzero = (dists_curr > 1e-5) & (dists_prev > 1e-5)
    if np.sum(nonzero) < 10:
        return None
        
    ratios = dists_prev[nonzero] / dists_curr[nonzero]
    scale = np.median(ratios)
    
    if scale > 10.0 or scale < 0.1:
        return None
        
    return scale


#get only 200 points for BA cus it takes too long otherwise
def filter_matches_BA(pts1, pts2, P, img_shape, max_points=200):
  if len(pts1) <= max_points:
      return pts1, pts2, P
  grid_side = int(math.sqrt(max_points))
  h, w = img_shape[:2]
  
  h_step = h / grid_side
  w_step = w / grid_side

  buckets = {}
    
  for i, pt in enumerate(pts1):
    x, y = pt
    row = int(y // h_step)
    col = int(x // w_step)
    
    if (row, col) not in buckets:
      buckets[(row, col)] = i

  keep_indices = list(buckets.values())

  #if we dont have enough, pick random points
  if len(keep_indices) < max_points:
    remaining = list(set(range(len(pts1))) - set(keep_indices))
    needed = max_points - len(keep_indices)
    if len(remaining) > 0:
      keep_indices.extend(np.random.choice(remaining, min(needed, len(remaining)), replace=False))

  keep_indices = np.array(keep_indices)
  return pts1[keep_indices], pts2[keep_indices], P[keep_indices]
