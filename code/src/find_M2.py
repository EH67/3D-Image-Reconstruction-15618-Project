""" 
Contains Serial & Parallel findM2 Algorithm (adapted from CMU's Computer Vision Course).

This calls the triangulation CUDA module to reconstruct the 3D points.
"""

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
import cuda_triangulation_module

def essentialMatrix(F, K1, K2):
  '''
  Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
  '''

  # ----- TODO -----
  ### BEGIN SOLUTION

  E_unconst = K2.T @ F @ K1
  U, S, Vh = np.linalg.svd(E_unconst)
  s = (S[0] + S[1]) / 2.0
  S_const = np.diag([s, s, 0.0])
  E = U @ S_const @ Vh
  if E[2, 2] != 0:
      E = E / E[2, 2]

  ### END SOLUTION
  return E

def camera2(E):
  """helper function to find the 4 possibile M2 matrices"""
  U,S,V = np.linalg.svd(E)
  m = S[:2].mean()
  E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
  U,S,V = np.linalg.svd(E)
  W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

  if np.linalg.det(U.dot(W).dot(V))<0:
      W = -W

  M2s = np.zeros([3,4,4])
  M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
  return M2s


def triangulate_cpu(C1, pts1, C2, pts2):
  '''
  Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
  Input:  C1, the 3x4 camera matrix
          pts1, the Nx2 matrix with the 2D image coordinates per row
          C2, the 3x4 camera matrix
          pts2, the Nx2 matrix with the 2D image coordinates per row
  Output: P, the Nx3 matrix with the corresponding 3D points per row
          err, the reprojection error.

  Hints:
  (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
  (2) Solve for the least square solution using np.linalg.svd
  (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from
      homogeneous coordinates to non-homogeneous ones)
  (4) Keep track of the 3D points and projection error, and continue to next point
  (5) You do not need to follow the exact procedure above.
  '''

  # ----- TODO -----
  ### BEGIN SOLUTION
  N = pts1.shape[0]
  P = np.zeros((N, 3))
  total_error = 0.0

  for i in range(N):
    p1 = pts1[i, :]
    p2 = pts2[i, :]

    A = np.zeros((4, 4))
    A[0, :] = p1[0] * C1[2, :] - C1[0, :]
    A[1, :] = p1[1] * C1[2, :] - C1[1, :]
    A[2, :] = p2[0] * C2[2, :] - C2[0, :]
    A[3, :] = p2[1] * C2[2, :] - C2[1, :]

    U, S, Vh = np.linalg.svd(A)
    P_homo = Vh[-1, :]

    P_non_homo = P_homo[0:3] / P_homo[3]
    P[i, :] = P_non_homo

    p1_proj_homo = C1 @ P_homo
    p1_proj = p1_proj_homo[0:2] / p1_proj_homo[2]
    p2_proj_homo = C2 @ P_homo
    p2_proj = p2_proj_homo[0:2] / p2_proj_homo[2]

    #error
    err1 = np.sum((p1_proj - p1)**2)
    err2 = np.sum((p2_proj - p2)**2)
    total_error += (err1 + err2)

  err = total_error

  ### END SOLUTION

  return P, err


def findM2_cpu(F, pts1, pts2, intrinsics):
    '''
    Q3.3: Function to find camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track
        of the projection error through best_error and retain the best one.
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'.

    '''

    K1, K2 = intrinsics['K1'], intrinsics['K2']

    # ----- TODO -----
    ### BEGIN SOLUTION
    E = essentialMatrix(F, K1, K2)

    M2s = camera2(E)
    M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    C1 = K1 @ M1

    best_error = float('inf')
    M2 = None
    C2 = None
    P = None

    N = pts1.shape[0]

    for i in range(4):
      M2_candidate = M2s[:, :, i]
      C2_candidate = K2 @ M2_candidate

      P_candidate, current_error = triangulate_cpu(C1, pts1, C2_candidate, pts2)

      z_coords_cam1 = P_candidate[:, 2]

      P_homo = np.hstack([P_candidate, np.ones((N, 1))])
      P_cam2 = (M2_candidate @ P_homo.T).T
      z_coords_cam2 = P_cam2[:, 2]

      if np.all(z_coords_cam1 >= 0) and np.all(z_coords_cam2 >= 0):
        if current_error < best_error:
          best_error = current_error
          M2 = M2_candidate
          C2 = C2_candidate
          P = P_candidate

    ### END SOLUTION

    return M2, C2, P


def findM2_cpu(F, pts1, pts2, intrinsics):
    if intrinsics is None:
        print("  [DEBUG] findM2 failed: 'intrinsics' is None.")
        return None, None, None

    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    C1 = K1 @ M1

    best_count = -1
    best_M2 = None
    best_C2 = None
    best_P = None
    best_i = -1

    N = pts1.shape[0]

    for i in range(4):
        M2_candidate = M2s[:, :, i]
        C2_candidate = K2 @ M2_candidate

        # --- DEBUGGING BLOCK ---
        try:
            P_candidate, _ = triangulate_cpu(C1, pts1, C2_candidate, pts2)
        except Exception as e:
            print(f"  [DEBUG] Candidate {i} triangulation crashed: {e}")
            continue

        if P_candidate is None:
             print(f"  [DEBUG] Candidate {i} triangulation returned None.")
             continue
        # -----------------------

        z_coords_cam1 = P_candidate[:, 2]
        P_homo = np.hstack([P_candidate, np.ones((N, 1))])
        P_cam2 = (M2_candidate @ P_homo.T).T
        z_coords_cam2 = P_cam2[:, 2]

        valid_indices = (z_coords_cam1 > 0) & (z_coords_cam2 > 0)
        count_valid = np.sum(valid_indices)

        # print(f"  [DEBUG] Cand {i}: {count_valid}/{N} points in front.")

        if count_valid > best_count:
            best_count = count_valid
            best_M2 = M2_candidate
            best_C2 = C2_candidate
            best_P = P_candidate
            best_i = i

    if best_M2 is None:
        print("  [DEBUG] findM2 failed: All 4 candidates failed triangulation or check.")
    # print(i)
    return best_M2, best_C2, best_P


def findM2_gpu(F, pts1, pts2, intrinsics):
    if intrinsics is None:
        print("  [DEBUG] findM2 failed: 'intrinsics' is None.")
        return None, None, None

    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F, K1, K2)
    M2s = camera2(E)
    M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    C1 = K1 @ M1

    best_count = -1
    best_M2 = None
    best_C2 = None
    best_P = None
    best_i = -1

    N = pts1.shape[0]

    for i in range(4):
        M2_candidate = M2s[:, :, i]
        C2_candidate = K2 @ M2_candidate

        # --- DEBUGGING BLOCK ---
        try:
            P_candidate = cuda_triangulation_module.triangulate(C1, pts1, C2_candidate, pts2)
        except Exception as e:
            print(f"  [DEBUG] Candidate {i} triangulation crashed: {e}")
            continue

        if P_candidate is None:
             print(f"  [DEBUG] Candidate {i} triangulation returned None.")
             continue
        # -----------------------

        z_coords_cam1 = P_candidate[:, 2]
        P_homo = np.hstack([P_candidate, np.ones((N, 1))])
        P_cam2 = (M2_candidate @ P_homo.T).T
        z_coords_cam2 = P_cam2[:, 2]

        valid_indices = (z_coords_cam1 > 0) & (z_coords_cam2 > 0)
        count_valid = np.sum(valid_indices)

        # print(f"  [DEBUG] Cand {i}: {count_valid}/{N} points in front.")

        if count_valid > best_count:
            best_count = count_valid
            best_M2 = M2_candidate
            best_C2 = C2_candidate
            best_P = P_candidate
            best_i = i

    if best_M2 is None:
        print("  [DEBUG] findM2 failed: All 4 candidates failed triangulation or check.")
    # print(i)
    return best_M2, best_C2, best_P