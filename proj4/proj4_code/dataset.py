#!/usr/bin/python3

"""Dataset loader."""

import copy
import os
import pickle
import random

from functools import reduce
from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.utils.data as data

from torch import nn

from proj4_code.part2b_patch import gen_patch
from proj4_code.utils import get_disparity

use_cuda = True and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
tensor_type = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
torch.set_default_tensor_type(tensor_type)
torch.backends.cudnn.deterministic = True
torch.manual_seed(
    333
)  # Do not change this, this is to ensure your result is reproducible


# See list of 23 scenes with GT here:
#   https://vision.middlebury.edu/stereo/data/scenes2014/
# 5 test scenes
TEST_SCENES = ["Adirondack", "Bicycle1", "Flowers", "Playroom", "Recycle"]
# 4 val scenes
VAL_SCENES = ["Jadeplant", "Motorcycle", "Piano", "Pipes"]
# 14 train scenes
TRAIN_SCENES = [
    "Playtable",
    "Shelves",
    "Vintage",
    "Backpack",
    "Cable",
    "Classroom1",
    "Couch",
    "Mask",
    "Shopvac",
    "Sticks",
    "Storage",
    "Sword1",
    "Sword2",
    "Umbrella",
]

data_name_to_idx = {
    "Adirondack": 0,
    "Backpack": 1,
    "Bicycle1": 2,
    "Cable": 3,
    "Classroom1": 4,
    "Couch": 5,
    "Flowers": 6,
    "Jadeplant": 7,
    "Mask": 8,
}
data_idx_to_name = {
    0: "Adirondack",
    1: "Backpack",
    2: "Bicycle1",
    3: "Cable",
    4: "Classroom1",
    5: "Couch",
    6: "Flowers",
    7: "Jadeplant",
    8: "Mask",
}


def loadbin(filename: str, device: torch.device = device) -> torch.Tensor:
    """Loads images or random idx from .bin file

    Args:
        filename: the name of the file
        device: torch.device

    Returns:
        x: torch.Tensor, left and right img in grayscale or random idx
    """
    with open(filename + ".dim") as file:
        dim = file.readlines()
        dim = np.array([int(x.strip()) for x in dim])
    size_1d = reduce(lambda x, y: x * y, dim)
    if os.path.exists(filename + ".type"):
        with open(filename + ".type") as file:
            type_ = file.readlines()
            type_ = [x.strip() for x in type_]
            assert len(type_) == 1
            type_ = type_[0]
    else:
        type_ = "float32"

    if type_ == "float32":
        x = torch.FloatTensor(torch.FloatStorage.from_file(filename, size=size_1d)).to(
            device
        )
    elif type_ == "int32":
        x = torch.IntTensor(torch.IntStorage.from_file(filename, size=size_1d)).to(
            device
        )
    elif type_ == "int64":
        x = torch.LongTensor(torch.LongStorage.from_file(filename, size=size_1d)).to(
            device
        )
    else:
        raise ValueError

    return x.reshape(tuple(dim))


class DataLoader(data.Dataset):
    """Data loader for the stereo dataset."""

    def __init__(
        self,
        image_dir: str,
        selected: List[str],
        batch_size: int = 512,
        ws: int = 11,
        use_cuda: bool = True,
        num_batches_per_epoch: int = 512,
    ) -> None:
        """
        DataLoader class constructor.

        Args:
            image_dir: path to the location where the images are stored
            selected: the indices of the images that are selected
            batch_size: the # of patch pairs to be loaded in a single batch
            ws: the size of the search window around the pixel
            use_cuda: Load the data on the GPU if True
            num_batches_per_epoch: The number of batches to be selected every epoch
        """
        self.imgs_left = dict()
        self.imgs_right = dict()
        self.disps = dict()
        self.batch_size = batch_size
        self.ws = ws
        self.int_keys = []
        self.nnz = dict()
        data_device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.num_batches_per_epoch = num_batches_per_epoch
        for name in selected:
            self.int_keys.append(name)
            self.imgs_left[name] = (
                torch.FloatTensor(np.load(image_dir + "/" + name + "_left.npy"))
                .unsqueeze(0)
                .to(data_device)
            )
            self.imgs_right[name] = (
                torch.FloatTensor(np.load(image_dir + "/" + name + "_right.npy"))
                .unsqueeze(0)
                .to(data_device)
            )
            # load disp imgs
            self.disps[name] = np.load(image_dir + "/" + name + "_disp.npy")

            # generate nnz
            disp = self.disps[name]
            y, x = np.nonzero(self.disps[name] != 0)
            self.nnz[name] = np.column_stack((y, x, disp[y, x]))

        self.count = np.arange(num_batches_per_epoch)

    def __len__(self) -> int:
        """" Computes the length of the dataset"""
        return self.num_batches_per_epoch
        # return len(self.disps)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to obtain a sample given its index, for the training or
        validation split.

        Official Reference:
            https://github.com/jzbontar/mc-cnn/blob/master/main.lua#L843

        Args:
            idx: index of the sample to be obtained

        Returns:
            x_batch: tensor of size (batch_size x 2, 1, ws, ws) representing a
                batch of patch pairs
            y_batch: tensor of size (batch_size) representing the ground truth
                matching cost for each pair in the batch (cost=0 for a
                positive/matching pair, and cost=1 for a negative pair)
        """
        x_batch = torch.zeros(self.batch_size * 2, 1, self.ws, self.ws).type(
            tensor_type
        )
        y_batch = torch.zeros(self.batch_size).type(tensor_type)

        _ = self.count[idx]

        d_neg_max = 18

        # get patches from img
        for i in range(1, self.batch_size // 2 + 1):
            d_pos = 0
            d_neg = np.random.randint(1, d_neg_max)
            if np.random.rand(1) < 0.5:
                d_neg = -d_neg

            ind = random.randint(0, len(self.nnz.keys()) - 1)
            img_name = list(self.nnz.keys())[ind]
            patch_idx = random.randint(0, len(self.nnz[img_name]) - 1)
            # print(self.nnz[img_name][patch_idx].shape)
            row, col, d = self.nnz[img_name][patch_idx]
            row = int(row)
            col = int(col)
            d = int(d)

            x0 = self.imgs_left[img_name]
            x1 = self.imgs_right[img_name]
            # Normalize image
            x0 = x0.add(-x0.mean()).div(x0.std())
            x1 = x1.add(-x1.mean()).div(x1.std())

            # Positive patch pair: get patch from left img and correctly
            #   horizontally-shifted patch from right img
            x_batch[i * 4 - 4] = gen_patch(x0, row, col, ws=self.ws)
            x_batch[i * 4 - 3] = gen_patch(x1, row, col - d + d_pos, ws=self.ws)

            # Negative patch pair: get patch from left img and randomly
            #   horizontally-shifted patch from right img
            x_batch[i * 4 - 2] = gen_patch(x0, row, col, ws=self.ws)
            x_batch[i * 4 - 1] = gen_patch(x1, row, col - d + d_neg, ws=self.ws)

            # Generate ground truth matching costs for both patch pairs (for
            # positive and negative patch)
            y_batch[i * 2 - 2] = 0
            y_batch[i * 2 - 1] = 1

        return x_batch, y_batch
