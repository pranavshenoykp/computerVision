import numpy as np


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_2d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    (N, _) = np.shape(points_2d)

    A = np.zeros([2*N, 11])
    b = np.zeros(2*N)
    row1 = np.zeros(11)
    row2 = np.zeros(11)

    for i in range(N):
        row1[:3] = points_3d[i,:]
        row1[3] = 1
        row1[4:8] = np.zeros(4)
        row1[8:] = points_2d[i,0] * points_3d[i,:] * -1

        row2[:4] = np.zeros(4)
        row2[4:7] = points_3d[i,:]
        row2[7] = 1
        row2[8:] = points_2d[i,1] * points_3d[i,:] * -1

        A[2*i,:] = row1
        A[2*i +1, :] = row2

        b[2*i] = points_2d[i,0]
        b[2*i+1] = points_2d[i,1]

    M_tmp = np.append(np.linalg.lstsq(A,b,rcond=None)[0],1)

    M = np.reshape(M_tmp,(3,4))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z,1] in homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 4 array of points [X_i,Y_i,Z_i,1] in homogeneous
            coordinates or n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    (n, d) = np.shape(points_3d)
    if d == 3:
        points_3d = np.hstack((points_3d, np.ones([n,1])))

    points_3d = points_3d.T
    projected_points_2d_hom = np.matmul(P,points_3d)


    projected_points_2d_hom = projected_points_2d_hom/projected_points_2d_hom[-1,:]

    projected_points_2d = projected_points_2d_hom[:-1,:]
    projected_points_2d = projected_points_2d.T


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Q = np.mat(M[:,:3])
    m4 = np.mat(M[:,3])

    cc = -1 * np.matmul(np.linalg.inv(Q), m4.T)

    cc = np.squeeze(np.array(cc))


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
