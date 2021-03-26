#!/usr/bin/python3

import numpy as np


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: float, Y: float, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    fvs = np.zeros([len(X), feature_width*feature_width])
    p_min = int(feature_width/2) -1
    p_max = int(feature_width/2) +1

    for i in range(len(X)):
        patch = image_bw[int(X[i] - p_min):int(X[i] + p_max), int(Y[i] - p_min):int(Y[i] + p_max)]
        fvs[i,:] = patch.reshape(1,256) / np.linalg.norm(patch)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
