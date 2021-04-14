"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    points = np.array(points, dtype=np.float32)

    cu = np.mean(points[:,0])
    cv = np.mean(points[:,1])

    su = 1. / np.std(points[:,0] - cu)
    sv = 1. / np.std(points[:,1] - cv)

    s = np.array([[su,0,0],[0,sv,0],[0,0,1]])
    c = np.array([[1,0,-1*cu],[0,1,-1*cv],[0,0,1]])

    T = np.matmul(s,c)

    points_normalized = np.zeros_like(points)
    points_normalized[:,0] = (points[:,0] - cu) * su
    points_normalized[:,1] = (points[:,1] - cv) * sv

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(
    F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = np.matmul(T_b.T, np.matmul(F_norm, T_a))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    (N, _) = np.shape(points_a)
    A = np.zeros([N,9])
    b = -1 * np.ones(N)

    points_a_norm, T_a = normalize_points(points_a)
    points_b_norm, T_b = normalize_points(points_b)

    # for i in range(N):
    #     (ua, va) = points_a_norm[i,:]
    #     (ub, vb) = points_b_norm[i,:]
    #     A[i,:] = np.array([ua*ub, va*ub, ub, ua*vb, vb*va, vb, ua, va])

    # # U, S, V = np.linalg.svd(A)

    # F_tmp = np.append(np.linalg.lstsq(A,b,rcond=None)[0],1)
    # F_tmp = np.reshape(F_tmp, (3,3))

    # U, S, V = np.linalg.svd(F_tmp)
    # S[2] = 0


    # F_norm = np.matmul(U, np.matmul(np.diag(S),V))


    for i in range(N):
        (ua, va) = points_a_norm[i,:]
        (ub, vb) = points_b_norm[i,:]
        A[i,:] = np.array([ua*ub, va*ub, ub, ua*vb, vb*va, vb, ua, va, 1])

    U, S, V = np.linalg.svd(A)

    F_tmp = V[-1].reshape(3,3)

    U,S,V = np.linalg.svd(F_tmp)
    S[2] = 0
    F_norm = np.dot(U,np.dot(np.diag(S),V))


    F = unnormalize_F(F_norm, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
