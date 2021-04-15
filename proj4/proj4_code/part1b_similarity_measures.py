#!/usr/bin/python3

"""
Defines different similarity measures used to compare blocks between two images
"""

import torch


def ssd_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> float:
    """Tests if two patches are similar by the SSD distance measure.

    SSD measure is sum of squared difference of pixel values in two patches. It
    is a good measure when the system has Gaussian noise.

    Args:
        patch1: one of the patch to compare (Tensor of any shape/dimensions)
        patch2: the other patch to compare (Tensor of the same shape as patch1)

    Returns:
        ssd_value: float representing a single ssd value of the patch
    """
    assert patch1.shape == patch2.shape
    ssd = 0  # placeholder

    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError(
        "`ssd_similarity_measure` function in "
        + "`part1b_similarity_measure.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return ssd


def sad_similarity_measure(patch1: torch.Tensor, patch2: torch.Tensor) -> float:
    """Tests if two patches are similar by the SAD distance measure.

    SAD is the sum of absolute difference. In general, absolute differences
    are more robust to large noise/outliers than squared differences.
    Ref: https://en.wikipedia.org/wiki/Sum_of_absolute_differences

    Args:
        patch1: one of the patch to compare (tensor of any shape/dimensions)
        patch2: the other patch to compare (tensor of the same shape as patch1)

    Returns:
        sad_value: float representing the scalar sad value of the patch
    """

    assert patch1.shape == patch2.shape
    sad = 0  # placeholder

    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError(
        "`sad_similarity_measure` function in "
        + "`part1b_similarity_measure.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return sad
