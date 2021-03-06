import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """
    Computes the mean and the standard deviation of all images present within
    the directory.

    Note: convert the image in grayscale and then in [0,1] before computing the
    mean and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = 1 / Variance

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None
    ############################################################################
    # Student code begin
    ############################################################################

    mean_array = []
    std_array = []

    for file in glob.glob(os.path.join(dir_name,"*","*","*.jpg")):
        img = Image.open(file)
        img = img.convert("L")
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        mean_array.append(np.mean(img))
        std_array.append(np.std(img))

    mean = np.mean(mean_array)
    std = np.mean(std_array)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
