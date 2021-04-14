import numpy as np
import math

from proj3_code.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    num_samples = np.log(1.0 - prob_success) / np.log(1 - (ind_prob_correct**sample_size))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def get_error_fundamental_matrix(
    point_a: np.ndarray, point_b: np.ndarray, F: np.ndarray) -> float:
    """
    Calculates the error function using the property of fundamental matrix:
    [u' v' 1] [F] [u v 1]' = 0

    """

    a = np.mat(np.append(point_a, 1.0))
    b = np.mat(np.append(point_b, 1.0))
    F = np.mat(F)

    err = float(np.abs(np.matmul(a, np.matmul(F, b.T))))

    return err




def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    sample_size = 8
    prob_success = 0.97
    ind_prob_correct = 0.5

    N_ransac = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)

    (N_matches, _) = np.shape(matches_a)

    N = min(N_ransac, N_matches)

    threshold =  0.005

    best_F = np.zeros([3,3])
    inliers_a = []
    inliers_b = []
    inliers_for_match_a = []
    inliers_for_match_b = []
    most_pts_in_thres = 0

    for ii in range(N):
        choices = np.random.choice(N, sample_size, replace=False)
        matches_a_selected = matches_a[choices]
        matches_b_selected = matches_b[choices]

        F_tmp = estimate_fundamental_matrix(matches_a_selected, matches_b_selected)

        sum_pts_in_thres = 0
        for jj in range(N_matches):
            if get_error_fundamental_matrix(matches_a[jj,:], matches_b[jj,:], F_tmp) < threshold:
                sum_pts_in_thres = sum_pts_in_thres + 1
                inliers_for_match_a.append(matches_a[jj,:])
                inliers_for_match_b.append(matches_b[jj,:])


        if sum_pts_in_thres > most_pts_in_thres:
            most_pts_in_thres = sum_pts_in_thres
            best_F = F_tmp
            inliers_a = np.array(inliers_for_match_a)
            inliers_b = np.array(inliers_for_match_b)

        inliers_for_match_a = []
        inliers_for_match_b = []
        sum_pts_in_thres = 0

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    if np.shape(inliers_a)[0] > 30:
        choices_inliers = np.random.choice(np.shape(inliers_a)[0], 30, replace=False)
        inliers_a = inliers_a[choices_inliers]
        inliers_b = inliers_b[choices_inliers]

    return best_F, inliers_a, inliers_b
