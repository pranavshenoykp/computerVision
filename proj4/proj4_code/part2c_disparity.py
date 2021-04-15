#!/usr/bin/python3

"""Calculates disparity maps."""

from typing import Callable

import numpy as np
import torch

from torch import nn


def calculate_mccnn_cost_volume(
    net: nn.Module,
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    block_size: int,
    sim_measure_function: Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    max_search_bound: int = 50,
) -> torch.Tensor:
    """
    Calculate the cost-volume at each pixel using MC-CNN by searching a
    small patch around a pixel from the left image in the right image.

    Note:
    1. It is important for this project to follow the convention of search
       input in left image and search target in right image

    2. The max_search_bound is defined from the patch center.

    3. This function will be similar to calculate_cost_volume from part1c,
       however, we will be computing similarity for a patch and a window
       instead of 2 patches

    4. To do so, stack all the right patches along the 0th dimension
       (reshaping if you have to) to create a (k, c, h, w) tensor,
       where k corresponds to the number of disparities you are considering.

       This is IMPORTANT as your runtime on colab will be extremely long
    if not implemented efficiently

    5. `sim_measure_function` has to be called once per pixel location under
        consideration,
       not once per disparity value. It also needs to be the
    `mc_cnn_similarity` function you implemented above

    Args:
        left_img: image Tensor of shape (C,H,W) from the left stereo camera.
            C will be >= 1.
        right_img: image Tensor of shape (C,H,W) from the right stereo camera
        block_size: the size of the block to be used for searching between the
            left and right images (should be odd)
        sim_measure_function: a function to measure similarity measure between
            a patch (c, h, w) and a list of patches as a Tensor (k, c, h, w)
        max_search_bound: the maximum horizontal distance (in terms of pixels)
            to use for searching

    Returns:
        cost_volume: The matching cost_volume for each disparity value at each
            pixel. Tensor of shape
            (H-2*(block_size//2), W-2*(block_size//2), max_search_bound)

    """
    assert left_img.shape == right_img.shape

    feature_extractor = net.conv
    fc_layers = net.classifier

    with torch.no_grad():
        cnn_left_img = feature_extractor(left_img.unsqueeze(0))
        cnn_right_img = feature_extractor(right_img.unsqueeze(0))

    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError(
        "`calculate_mccnn_cost_volume` function in "
        + "`part2c_disparity.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return cost_volume


def calculate_mccnn_disparity_map(
    net: nn.Module,
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    block_size: int,
    sim_measure_function: Callable[
        [torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    max_search_bound: int = 50,
) -> torch.Tensor:
    """Calculate the disparity value at each pixel using MC-CNN by searching a
    small patch around a pixel from the left image in the right image.

    Hint: Don't repeat yourself, use the cost_volume you'll calculate below

    Note:
        1. While searching for disparity value for a patch, it may happen that
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
        2. The max_search_bound is defined from the patch center.
        3. This function will be similar to calculate_disparity_map from
        part 1c. However, we will be computing similarity for a patch and a
        window instead of 2 patches.

    Args:
        left_img: image Tensor of shape (C,H,W) from the left stereo camera.
            C will be >= 1.
        right_img: image Tensor of shape (C,H,W) from the right stereo camera
        block_size: the size of the block to be used for searching between the
            left and right images (should be odd)
        sim_measure_function: a function to measure similarity measure between
            a patch (c, h, w) and a list of patches as a Tensor (k, c, h, w)
        max_search_bound: the maximum horizontal distance (in terms of pixels)
            to use for searching

    Returns:
        disparity_map: The map of disparity values at each pixel. Tensor of
            shape (H-2*(block_size//2), W-2*(block_size//2))

    """
    cost_volume = calculate_mccnn_cost_volume(
        net,
        left_img,
        right_img,
        block_size=block_size,
        sim_measure_function=sim_measure_function,
        max_search_bound=max_search_bound,
    )
    return torch.argmin(cost_volume, dim=-1)


def mc_cnn_similarity(
    fc_layers: nn.Sequential,
    cnn_patch: torch.Tensor,
    cnn_search_window: torch.Tensor
) -> torch.Tensor:
    """
    Computes similarity between the CNN features of an arbitrary (1, C, H, W)
    patch and a window of (k, C, H, W) patch features joined together at the
    0th dimension using the fully-connected layers of the MC-CNN.

    IMPORTANT: You should not use a loop for anything other
    than constructing the tensors to be sent to the fc-layers,
    or your runtime will be several hours on colab even with GPU.

    Steps to implement:

    1. For every tensor at the 0th dimension of the search window,
        you need to join it with the patch from the left image at the 0th
        dimensions, then completely flatten it to form a batch of tensors to be
        sent to the fc_layers.
    2. To do this, sample the patch, and the i-th tensor of the 0th dimension
       of the window. Create a new 0th dimension for both tensors, and
       concatenate them at that. Then flatten this concatenation with:
            concatenated_patches.view(-1)
    3. Stack the flattened tensors at a new 0th dimension to form a batch

    You should have to call the fc_layers only once if you've implemented it
    correctly.

    Args:
        fc_layers: The fully-connected layers of the model to compute
            similarity with
        cnn_patch: a patch of shape (1,C,H,W) (from the left_image) to compute
            the similarity for
        cnn_search_window: a batch of patches of shape (k,C,H,W) (from the
            right image) to compute the similarity with

    Returns:
        cnn_similarity: a Tensor of shape (k,1) containing a set of similarity
            values for each patch in the search window
    """
    ###########################################################################
    # Student code begins
    ###########################################################################

    # IMPORTANT: Read the docstring _carefully_ before you write this code

    ###########################################################################

    raise NotImplementedError(
        "`mc_cnn_similarity` function in "
        + "`part2c_disparity.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return cnn_similarity
