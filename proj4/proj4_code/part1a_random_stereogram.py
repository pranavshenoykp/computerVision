#!/usr/bin/python3

"""Generates random stereogram."""

from typing import Any, List, Tuple

import numpy as np
import torch


def generate_random_stereogram(
    im_size: Tuple[int, int, int] = (51, 51, 3), disparity: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random stereogram for the given image size.

    The block which is shifted is centered at the image center and is of 0.5
    times the dimension of the input.

    Steps:
    1. Generate the left image to be random 0/1 values
    2. Set the right image as the copy of the left image
    3. Move a block around the center block in the right image by
       'disparity val' to the left
    4. Fill in the 'hole' in the right image with random values

    Note:
    1. The block to be moved is a square of size (H//2,W//2) at the center
       pixel of the image (H,W,C). Note the use of integer division.
    2. The values in the images should be 0 and 1 (at random)
    3. Your code will not be tested with inputs where moving the block with the
       given disparity takes the block out of bounds.
    4. The resulting image should be grayscale, i.e. a pixel value should be
       same in all the channels. image[x,y,0] == image[x,y,1] == ..... and so on
       for all the channels

    Args:
       im_size: The size of the image to be be generated
       disparity: the shift to be induced in the right image

    Returns:
       im_left: tensor of shape (H,W,C) representing the left image
       im_right: tensor of shape (H,W,C) representing the right image
    """
    H, W, C = im_size
    block_size = (H // 2, W // 2)
    im_left = torch.zeros(1)  # placeholder, not actual size
    im_right = torch.zeros(1)  # placeholder, not actual size

    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError(
        "`generate_random_stereogram` function in "
        + "`part1a_random_stereogram.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return im_left, im_right
