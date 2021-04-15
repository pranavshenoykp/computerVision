#!/usr/bin/python3

"""Generates a patch from an image."""

import torch


def gen_patch(image: torch.Tensor, x: int, y: int, ws: int = 11) -> torch.Tensor:
    """Returns a patch at a specific location of the image.

    x, y in this case is a top left corner of the patch, for example if (x,y)
    is (0,0) you should return a patch over (0,0) and (ws,ws)

    For corner case, you can pad the output with zeros such that we always have
    (channel, ws, ws) dimension output

    Args:
        image: image of type Tensor with dimension (C, width, height)
        x: x coordinate in the image
        y: y coordinate in the image
        ws: window size or block size of the patch we want

    Returns:
        patch: a patch of size (C, ws, ws) of type Tensor
    """
    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError(
        "`gen_patch` function in " + "`part2b_patch.py` needs to be implemented"
    )

    ###########################################################################
    # Student code ends
    ###########################################################################
    return patch
