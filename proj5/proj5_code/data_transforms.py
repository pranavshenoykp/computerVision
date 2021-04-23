"""
Contains functions with different data transforms
"""

from typing import Tuple, Sequence

import numpy as np
import torchvision.transforms as transforms


def get_fundamental_transforms(
    inp_size: Tuple[int, int]) -> transforms.Compose:
    """
    Returns the core transforms necessary to feed the images to our model.

    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fundamental_transforms: transforms.Compose with the fundamental
            transforms
    """
    fundamental_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError('`get_fundamental_transforms` function in '
        + '`data_transforms.py` needs to be implemented')

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]) -> transforms.Compose:
    """
    Returns the core transforms in addition to augmentation.

    A few uggestions: Jittering, Flipping, Cropping, Rotating.
    Args:
        inp_size: tuple denoting the dimensions for input to the model

    Returns:
        fund_aug_transforms: transforms.Compose with fundamental and
            augmentation transforms
    """
    fund_aug_transforms = None
    ###########################################################################
    # Student code begin
    ###########################################################################

    raise NotImplementedError('`get_fundamental_augmentation_transforms` '
        + 'function in `data_transforms.py` needs to be implemented')

    ###########################################################################
    # Student code end
    ###########################################################################
    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float]
) -> transforms.Compose:
    """
    Returns the core transforms in addition to normalization.

    These transforms will be applied to the validation set because we don't
    want to augment them, but we still want these other basic transformations.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the raw
            dataset

    Returns:
        fund_norm_transforms: transforms.Compose with the fundamental
            and normalization transforms
    """
    fund_norm_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError('`get_fundamental_normalization_transforms` '
        + 'function in `data_transforms.py` needs to be implemented')

    ###########################################################################
    # Student code ends
    ###########################################################################
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int],
    pixel_mean: Sequence[float],
    pixel_std: Sequence[float]
) -> transforms.Compose:
    """
    Returns the core, augmentation, and normalization transforms.

    These transforms will be applied to the training set.

    Args:
        inp_size: tuple denoting the dimensions for input to the model
        pixel_mean: image channel means, over all images of the raw dataset
        pixel_std: image channel standard deviations, for all images of the
            raw dataset

    Returns:
        all_transforms: transforms.Compose with all the transforms
    """
    all_transforms = None
    ###########################################################################
    # Student code begins
    ###########################################################################

    raise NotImplementedError('`get_all_transforms` function in '
        + '`data_transforms.py` needs to be implemented')

    ###########################################################################
    # Student code ends
    ###########################################################################
    return all_transforms
