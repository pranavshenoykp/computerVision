#!/usr/bin/python3

from functools import reduce
import copy
import numpy as np
import PIL
import pickle
import torch
from torch import nn
import torch.utils.data as data
import random
import os

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from typing import Any, Callable, List, Tuple

from cv2 import resize

use_cuda = True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(tensor_type)

torch.backends.cudnn.deterministic = True
torch.manual_seed(
    333
)  # do not change this, this is to ensure your result is reproduciable

from proj4_code.part1c_disparity_map import calculate_disparity_map
from proj4_code.part1b_similarity_measures import (
    ssd_similarity_measure,
    sad_similarity_measure,
)

"""
File adapted from project 2
"""


def stereo_helper_fn(im_left, im_right, block_size=[5, 9, 13], max_search_bound=15):
    """
    This helper function will help us in calculating disparity maps for different parameters.
    It also plots the image.

    Please tune the parameters and see the effect of them for different inputs.

    Args:
      - im_left: the left image
      - im_right: the right image
      - block_size: list of different block sizes to be used
      - max_search_bound: the max horizontal displacement to look for the most similar patch
                          (Refer to the project webpage for more details)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

    ax1.imshow(im_left.data.cpu().numpy(), interpolation=None)
    ax1.title.set_text("Left image")
    ax1.autoscale(False)
    ax1.set_axis_off()

    ax2.imshow(im_right.data.cpu().numpy(), interpolation=None)
    ax2.title.set_text("Right image")
    ax2.autoscale(False)
    ax2.set_axis_off()

    plt.show()

    # fig, ax = plt.subplots(len(block_size),2, figsize=(15, 10*len(block_size)))

    for idx, block in enumerate(block_size):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 20))
        # **calculate the disparity maps**

        # Using SAD similarity function
        disp_map_sad = calculate_disparity_map(
            im_left,
            im_right,
            block_size=int(block),
            sim_measure_function=sad_similarity_measure,
            max_search_bound=max_search_bound,
        )

        # Using SSD similarity function
        disp_map_ssd = calculate_disparity_map(
            im_left,
            im_right,
            block_size=block,
            sim_measure_function=ssd_similarity_measure,
            max_search_bound=max_search_bound,
        )

        im = ax1.imshow(disp_map_sad.data.cpu().numpy(), cmap="jet")
        ax1.set_title("Disparity Map - SAD ({}x{} patch)".format(block, block))
        ax1.autoscale(True)
        ax1.set_axis_off()
        # cbar = fig.colorbar(im, ax=ax1, cmap='jet', shrink=0.3)
        cbar = fig.colorbar(im, ax=ax1, shrink=0.3)

        im = ax2.imshow(disp_map_ssd.data.cpu().numpy(), cmap="jet")
        ax2.set_title("Disparity Map - SSD ({}x{} patch)".format(block, block))
        ax2.autoscale(True)
        ax2.set_axis_off()
        # cbar = fig.colorbar(im, ax=ax2, cmap='jet', shrink=0.3)
        cbar = fig.colorbar(im, ax=ax2, shrink=0.3)

        plt.show()


def rgb2gray(img: np.ndarray) -> np.ndarray:
    """
    Use the coefficients used in OpenCV, found here:
    https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

    Args:
    -   Numpy array of shape (M,N,3) representing RGB image

    Returns:
    -   Numpy array of shape (M,N) representing grayscale image
    """
    # Grayscale coefficients
    c = [0.299, 0.587, 0.114]
    return img[:, :, 0] * c[0] + img[:, :, 1] * c[1] + img[:, :, 2] * c[2]


def PIL_resize(img: np.ndarray, ratio: Tuple[float, float]) -> np.ndarray:
    """
    Args:
    - img: Array representing an image
    - size: Tuple representing new desired (width, height)

    Returns:
    - img
    """
    H, W, _ = img.shape
    img = numpy_arr_to_PIL_image(img, scale_to_255=True)
    img = img.resize((int(W * ratio[1]), int(H * ratio[0])), PIL.Image.LANCZOS)
    img = PIL_image_to_numpy_arr(img)
    return img


def PIL_image_to_numpy_arr(img, downscale_by_255=True):
    """
    Args:
    - img
    - downscale_by_255

    Returns:
    - img
    """
    img = np.asarray(img)
    img = img.astype(np.float32)
    if downscale_by_255:
        img /= 255
    return img


def im2single(im: np.ndarray) -> np.ndarray:
    """
    Args:
    - img: uint8 array of shape (m,n,c) or (m,n) and in range [0,255]

    Returns:
    - im: float or double array of identical shape and in range [0,1]
    """
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    """
    Args:
    - im: float or double array of shape (m,n,c) or (m,n) and in range [0,1]

    Returns:
    - im: uint8 array of identical shape and in range [0,255]
    """
    im *= 255
    im = im.astype(np.uint8)
    return im


def numpy_arr_to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    """
    Args:
    - img: in [0,1]

    Returns:
    - img in [0,255]

    """
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def load_image(path: str) -> np.ndarray:
    """
    Args:
    - path: string representing a file path to an image

    Returns:
    - float or double array of shape (m,n,c) or (m,n) and in range [0,1],
      representing an RGB image
    """
    img = PIL.Image.open(path)
    img = np.asarray(img)
    float_img_rgb = im2single(img)
    return float_img_rgb


def save_image(path: str, im: np.ndarray) -> bool:
    """
    Args:
    - path: string representing a file path to an image
    - img: numpy array

    Returns:
    - retval indicating write success
    """
    img = copy.deepcopy(im)
    img = single2im(img)
    pil_img = numpy_arr_to_PIL_image(img, scale_to_255=False)
    return pil_img.save(path)


def write_objects_to_file(fpath: str, obj_list: List[Any]):
    """
    If the list contents are float or int, convert them to strings.
    Separate with carriage return.

    Args:
    - fpath: string representing path to a file
    - obj_list: List of strings, floats, or integers to be written out to a file, one per line.

    Returns:
    - None
    """
    obj_list = [str(obj) + "\n" for obj in obj_list]
    with open(fpath, "w") as f:
        f.writelines(obj_list)


def hstack_images(img1, img2):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """

    # CHANGED
    imgA = np.array(img1)
    imgB = np.array(img2)
    Height = max(imgA.shape[0], imgB.shape[0])
    Width = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[: imgA.shape[0], : imgA.shape[1], :] = imgA
    newImg[: imgB.shape[0], imgA.shape[1] :, :] = imgB

    # newImg = PIL.Image.fromarray(np.uint8(newImg))
    return newImg


def generate_delta_fn_images(im_size):
    """
    Generates a pair of left and right (stereo pair) images of a single point.
    This point mimics a delta function and will manifest as a single pixel
    on the same vertical level in both the images. The horizontal distance
    between the pixels will be proportial to the 3D depth of the image
    """

    H = im_size[0]
    W = im_size[1]

    im1 = torch.zeros((H, W, 3))
    im2 = torch.zeros((H, W, 3))

    # pick a location of a pixel in im1 randomly
    im1_r = random.randint(0, H - 1)
    im1_c = random.randint(W // 2, W - W // 4)

    im1[im1_r, im1_c, :] = torch.FloatTensor([1.0, 1.0, 1.0])

    # pick a location of the pixel in im2
    im2_r = im1_r
    im2_c = im1_c - random.randint(1, W // 4 - 1)

    im2[im2_r, im2_c, :] = torch.FloatTensor([1.0, 1.0, 1.0])

    return im1, im2


def verify(function: Callable) -> str:
    """Will indicate with a print statement whether assertions passed or failed
    within function argument call.
    Args:
    - function: Python function object
    Returns:
    - string
    """
    try:
        function()
        return '\x1b[32m"Correct"\x1b[0m'
    except (AssertionError, RuntimeError) as e:
        print(e)
        return '\x1b[31m"Wrong"\x1b[0m'


def save_model(network: nn.Module, path: str) -> None:
    torch.save(network.state_dict(), path)


def load_model(network: nn.Module, path: str, device="cpu", strict=True) -> nn.Module:
    network.load_state_dict(
        torch.load(path, map_location=torch.device(device)), strict=strict
    )
    return network


def get_disparity(nnz: torch.Tensor, ind: int) -> Tuple[int, int, int, int]:
    """Get the img idx by int, and get the row and col information stored in nnz.
    The row/col is not fully generated randomly because we don't want to patch position to be
    fully random, for example, if row1=3 and col1=3, in the next random pick, row2=4 and col2=4, then
    these two patches can be very similar which will require much much more sample iterations. So here
    in nnz, we generated the ramdom patch positions and make sure those patches will not have much overlap.
        Args:
        - nnz: a nx4 size array, n is a extremely large number, for each row which is 1x4, the first number is the idx of the img,
        the second and third number is the position of random patch's row and col idx, the fourth number is a vertical shift of this patch.
        - ind: int, a random number to pick in nnz
        Returns:
        - img: int, the index of which img pairs to pick
        - dim3: int, the row idx of the random patch
        - dim4: int, the col idx of the random patch
        - d: int, a vertical shift
    """
    img = nnz[ind, 0]
    dim3 = nnz[ind, 1]
    dim4 = nnz[ind, 2]
    d = nnz[ind, 3]
    return img, dim3, dim4, d


def save_model_for_evaluation(model: nn.Module) -> None:
    """Saves the final trained model for evaluation on Gradescope

    Store the final saved model in your proj4_code folder

    Args:
    -   model: The final trained model to upload for evaluation
               It must match the MCNET model that you submit
    """
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    class_name = model.__class__.__name__
    saved_model_data = {
        "state_dict": state_dict,
        "model_class": class_name,
        "window_size": model.ws,
    }
    torch.save(saved_model_data, "./final_model_data.pth")


def evaluate_stereo(
    gt: np.ndarray, disp: np.ndarray, max_disp: float = 280
) -> Tuple[float, float, float, float]:
    """Evaluates stereo performance metrics for a disparity map

    Args:
    -   gt: Ground truth disparity map
    -   disp: Disparity map to be evaluated
    -   max_disparity: Maximum disparity value to be evaluated
                       (required to clip spurious values)

    Returns:
    -   avg: Average error of the disparity map
    -   bad1: percentage of pixels with error greater than 1px
    -   bad2: percentage of pixels with error greater than 2px
    -   bad4: percentage of pixels with error greater than 4px
    """
    mask = gt != np.inf

    if disp.shape != gt.shape:
        ratio = float(gt.shape[1]) / disp.shape[1]
        # Scaling the disparity map means we need to scale disparities as well
        disp = resize(disp, (gt.shape[0], gt.shape[1])).transpose() * ratio
    disp[disp > max_disp] = max_disp

    errmap = np.abs(gt - disp) * mask
    avgerr = errmap[mask].mean()
    bad1map = (errmap > 1) * mask
    bad1 = (
        bad1map[mask].sum() / float(mask.sum()) * 100
    )  # percentage of bad pixels whose error is greater than 1

    bad2map = (errmap > 2) * mask
    bad2 = (
        bad2map[mask].sum() / float(mask.sum()) * 100
    )  # percentage of bad pixels whose error is greater than 2

    bad4map = (errmap > 4) * mask
    bad4 = (
        bad4map[mask].sum() / float(mask.sum()) * 100
    )  # percentage of bad pixels whose error is greater than 4

    return avgerr, bad1, bad2, bad4
