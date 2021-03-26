#!/usr/bin/python3

import copy
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import torch

from proj2_code.part1_harris_corner import compute_image_gradients
from torch import nn
from typing import Tuple


"""
Authors: Vijay Upadhya, John Lambert, Cusuh Ham, Patsorn Sangkloy, Samarth
Brahmbhatt, Frank Dellaert, James Hays, January 2021.

Implement SIFT  (See Szeliski 7.1.2 or the original publications here:
    https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

Your implementation will not exactly match the SIFT reference. For example,
we will be excluding scale and rotation invariance.

You do not need to perform the interpolation in which each gradient
measurement contributes to multiple orientation bins in multiple cells. 
"""


def get_magnitudes_and_orientations(
    Ix: np.ndarray,
    Iy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will return the magnitudes and orientations of the
    gradients at each pixel location.

    Args:
        Ix: array of shape (m,n), representing x gradients in the image
        Iy: array of shape (m,n), representing y gradients in the image
    Returns:
        magnitudes: A numpy array of shape (m,n), representing magnitudes of
            the gradients at each pixel location
        orientations: A numpy array of shape (m,n), representing angles of
            the gradients at each pixel location. angles should range from
            -PI to PI.
    """
    magnitudes = []  # placeholder
    orientations = []  # placeholder

    magnitudes = np.sqrt(Ix * Ix + Iy * Iy)
    orientations = np.arctan2(Iy, Ix)


    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################



    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return magnitudes, orientations

def simplify_blocks(feature: np.ndarray, nrows: int=4, ncols: int=4) -> np.ndarray:
    """
    Converts a 16x16 patch to a 4x4x4 patch and then into a sorted 16x16 patch
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29, 30, 31],
       [32, 33, 34, 35, 36, 37, 38, 39],
       [40, 41, 42, 43, 44, 45, 46, 47],
       [48, 49, 50, 51, 52, 53, 54, 55],
       [56, 57, 58, 59, 60, 61, 62, 63]])

    to 

    array([[ 0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
       [ 4,  5,  6,  7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31],
       [32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59],
       [36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63]])
    """
    h,w = feature.shape
    tmp = feature.reshape(h//nrows,nrows,-1,ncols).swapaxes(1,2).reshape(-1, nrows,ncols)
    return (tmp.ravel().reshape(-1,nrows*ncols))

def get_gradient_histogram_vec_from_patch(
    window_magnitudes: np.ndarray,
    window_orientations: np.ndarray
) -> np.ndarray:
    """ Given 16x16 patch, form a 128-d vector of gradient histograms.

    Key properties to implement:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the terminology
        used in the feature literature to describe the spatial bins where
        gradient distributions will be described. The grid will extend
        feature_width/2 to the left of the "center", and feature_width/2 - 1 to
        the right
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions. The bin centers for the histogram
        should be at -7pi/8,-5pi/8,...5pi/8,7pi/8. The histograms should be
        added to the feature vector left to right then row by row (reading
        order).

    Do not normalize the histogram here to unit norm -- preserve the histogram
    values. A useful function to look at would be np.histogram.

    Args:
        window_magnitudes: (16,16) array representing gradient magnitudes of the
            patch
        window_orientations: (16,16) array representing gradient orientations of
            the patch

    Returns:
        wgh: (128,1) representing weighted gradient histograms for all 16
            neighborhoods of size 4x4 px
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    bins = [-np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    window_magnitudes = simplify_blocks(window_magnitudes)
    window_orientations = simplify_blocks(window_orientations)

    feature_vector = np.zeros([16,8])
    for i in range(16):
        feature_vector[i,:] = np.histogram(window_orientations[i], bins=bins, weights = window_magnitudes[i])[0]
        
    wgh = feature_vector.reshape(128,1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return wgh


def get_feat_vec(
    x: float,
    y: float,
    magnitudes,
    orientations,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the feature vector for a specific interest point. To
    start with, you might want to simply use normalized patches as your local
    feature. This is very simple to code and works OK. However, to get full
    credit you will need to implement the more effective SIFT descriptor (see
    Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/). Your implementation does not need
    to exactly match the SIFT reference.


    Your (baseline) descriptor should have:
    (1) Each feature should be normalized to unit length.
    (2) Each feature should be raised to the 1/2 power, i.e., square-root SIFT
        (read https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)

    For our tests, you do not need to perform the interpolation in which each
    gradient measurement contributes to multiple orientation bins in multiple
    cells. As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest orientation
    bins within each cell, for 8 total contributions. The autograder will only
    check for each gradient contributing to a single bin.

    Args:
        x: a float, the x-coordinate of the interest point
        y: A float, the y-coordinate of the interest point
        magnitudes: A numpy array of shape (m,n), representing image gradients
            at each pixel location
        orientations: A numpy array of shape (m,n), representing gradient
            orientations at each pixel location
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fv: A numpy array of shape (feat_dim,1) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g., 128 for standard
            SIFT). These are the computed features.
    """

    fv = []  # placeholder

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`get_feat_vec` function in ' +
        '`student_sift.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fv


def get_SIFT_descriptors(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    feature_width: int = 16
) -> np.ndarray:
    """
    This function returns the 128-d SIFT features computed at each of the input
    points. Implement the more effective SIFT descriptor (see Szeliski 4.1.2 or
    the original publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Args:
        image: A numpy array of shape (m,n), the image
        X: A numpy array of shape (k,), the x-coordinates of interest points
        Y: A numpy array of shape (k,), the y-coordinates of interest points
        feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e.,
            every cell of your local SIFT-like feature will have an integer
            width and height). This is the initial window size we examine
            around each keypoint.
    Returns:
        fvs: A numpy array of shape (k, feat_dim) representing all feature
            vectors. "feat_dim" is the feature_dimensionality (e.g., 128 for
            standard SIFT). These are the computed features.
    """
    assert image_bw.ndim == 2, 'Image must be grayscale'

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`get_SIFT_descriptors` function in ' +
        '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return fvs


### ----------------- OPTIONAL (below) ------------------------------------

## Implementation of the function below is  optional (extra credit)

def get_sift_features_vectorized(
    image_bw: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray
) -> np.ndarray:
    """
    This function is a vectorized version of `get_SIFT_descriptors`.

    As before, start by computing the image gradients, as done before. Then
    using PyTorch convolution with the appropriate weights, create an output
    with 10 channels, where the first 8 represent cosine values of angles
    between unit circle basis vectors and image gradient vectors at every
    pixel. The last two channels will represent the (dx, dy) coordinates of the
    image gradient at this pixel. The gradient at each pixel can be projected
    onto 8 basis vectors around the unit circle

    Next, the weighted histogram can be created by element-wise multiplication
    of a 4d gradient magnitude tensor, and a 4d gradient binary occupancy
    tensor, where a tensor cell is activated if its value represents the
    maximum channel value within a "fibre" (see
    http://cs231n.github.io/convolutional-networks/ for an explanation of a
    "fibre"). There will be a fibre (consisting of all channels) at each of the
    (M,N) pixels of the "feature map".

    The four dimensions represent (N,C,H,W) for batch dim, channel dim, height
    dim, and weight dim, respectively. Our batch size will be 1.

    In order to create the 4d binary occupancy tensor, you may wish to index in
    at many values simultaneously in the 4d tensor, and read or write to each
    of them simultaneously. This can be done by passing a 1D PyTorch Tensor for
    every dimension, e.g., by following the syntax:
        My4dTensor[dim0_idxs, dim1_idxs, dim2_idxs, dim3_idxs] = 1d_tensor.

    Finally, given 8d feature vectors at each pixel, the features should be
    accumulated over 4x4 subgrids using PyTorch convolution.

    You may find torch.argmax(), torch.zeros_like(), torch.meshgrid(),
    flatten(), torch.arange(), torch.unsqueeze(), torch.mul(), and
    torch.norm() helpful.

    Returns:
        fvs
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    raise NotImplementedError('`get_SIFT_features_vectorized` function in ' +
        '`part4_sift_descriptor.py` needs to be implemented')

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs