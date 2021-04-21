#!/usr/bin/python3

"""Calculates disparity maps."""

from typing import Callable, Tuple

import numpy as np
import torch


def calculate_disparity_map(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    block_size: int,
    sim_measure_function: Callable,
    max_search_bound: int = 50,
) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small patch
    around a pixel from the left image in the right image.

    Note:
    1. It is important for this project to follow the convention of search
       input in left image and search target in right image
    2. While searching for disparity value for a patch, it may happen that
       there are multiple disparity values with the minimum value of the
       similarity measure. In that case we need to pick the smallest disparity
       value. Please check the numpy's argmin and pytorch's argmin carefully.
       Example:
       -- diparity_val -- | -- similarity error --
       -- 0               | 5
       -- 1               | 4
       -- 2               | 7
       -- 3               | 4
       -- 4               | 12

       In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
        left_img: image Tensor of shape (H,W,C) from the left stereo camera.
            C will be >= 1
        right_img: image Tensor of shape (H,W,C) from the right stereo camera
        block_size: the size of the block to be used for searching between the
            left and right image (should be odd)
        sim_measure_function: a function to measure similarity measure between
            two tensors of the same shape; returns the error value
        max_search_bound: the maximum horizontal distance (in terms of pixels)
            to use for searching
    Returns:
        disparity_map: The map of disparity values at each pixel. Tensor of
            shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape
    (H,W,C) = left_img.shape
    H_offset = block_size//2
    W_offset = block_size//2
    disp_map = torch.zeros(H-2*H_offset,W-2*W_offset)  # placeholder, this is not the actual size


    ###########################################################################
    # Student code begins
    ###########################################################################

    for ii in range(H-2*H_offset):
        for jj in range(W-2*W_offset):
            left_patch = left_img[ii:ii+block_size,jj:jj+block_size, :]
            similarity_error_array = -1 * np.ones(max_search_bound)
            
            for kk in range(max_search_bound):
                jj_start = max(jj - kk, 0)
                jj_end = max(jj - kk + block_size, block_size)

                right_patch = right_img[ii:ii+block_size, jj_start:jj_end, :]
                similarity_error_array[kk] = sim_measure_function(left_patch, right_patch)

            disp_map[ii,jj] = np.argmin(similarity_error_array)

    ###########################################################################
    # Student code ends
    ###########################################################################
    return disp_map


def calculate_cost_volume(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    max_disparity: int,
    sim_measure_function: Callable,
    block_size: int = 9,
):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note:
    1. It is important for this project to follow the convention of search
       input in left image and search target in right image
    2. If the shifted patch in the right image will go out of bounds, it is
       good to set the default cost for that pixel and disparity to be
       something high (we recommend 255) so that when we consider costs, valid
       disparities will have a lower cost.

    Args:
        left_img: image Tensor of shape (H,W,C) from the left stereo camera.
            C will be 1 or 3.
        right_img: image Tensor of shape (H,W,C) from the right stereo camera
        max_disparity: represents the range of disparity values we will
            consider (0 to max_disparity-1)
        sim_measure_function: a function to measure similarity measure between
            two tensors of the same shape; returns the error value
        block_size: the size of the block to be used for searching between the
            left and right image, it should be odd
    Returns:
        cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
            dimensions, and D is max_disparity. cost_volume[x,y,d] represents
            the similarity or cost between a patch around left[x,y] and a patch
            shifted by disparity d in the right image.
    """
    # placeholders
    H = left_img.shape[0]
    W = right_img.shape[1]
    H_offset = block_size//2
    W_offset = block_size//2
    cost_volume = torch.ones(H, W, max_disparity) * 255

    ###########################################################################
    # Student code begins
    ###########################################################################

    for ii in range(H-2*H_offset):
        for jj in range(W-2*W_offset):
            left_patch = left_img[ii:ii+block_size,jj:jj+block_size, :]
            similarity_error_array = np.ones(max_disparity)
            
            for kk in range(max_disparity):
                if jj - kk >= 0:
                    jj_start = max(jj - kk, 0)
                    jj_end = max(jj - kk + block_size, block_size)

                    right_patch = right_img[ii:ii+block_size, jj_start:jj_end, :]
                    similarity_error_array[kk] = sim_measure_function(left_patch, right_patch)
                else:
                    similarity_error_array[kk] = 255.0

            cost_volume[ii+H_offset,jj+W_offset,:] = torch.tensor(similarity_error_array)

    ###########################################################################
    # Student code ends
    ###########################################################################

    return cost_volume
