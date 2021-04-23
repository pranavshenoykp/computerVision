"""
Script with Pytorch's dataloader class
"""

import os
import glob
from typing import Dict, List, Tuple

import torch
import torchvision
from PIL import Image
import torch.utils.data as data


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """
        Constructor for the ImageLoader class that sets `curr_folder` for the
        corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """
        Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the
                classes in this dataset

        Returns:
            img_paths: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)
        #######################################################################
        # Student code begins
        #######################################################################

        raise NotImplementedError('`load_imagepaths_with_labels` function in '
            + '`image_loader.py` needs to be implemented')

        #######################################################################
        # Student code ends
        #######################################################################
        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """
        Gets the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical
        order, i.e., if your classes are
            ['apple', 'giraffe', 'elephant', 'cat'],
        then the class labels dictionary should be:
            {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            classes: dict of class names (string) to integer labels
        """

        classes = dict()
        #######################################################################
        # Student code begins
        #######################################################################

        raise NotImplementedError('`get_classes` function in '
            + '`dl_utils.py` needs to be implemented')

        #######################################################################
        # Student code ends
        #######################################################################
        return classes

    def load_img_from_path(self, path: str) -> Image:
        """
        Loads an image as grayscale (using Pillow).

        Note: Do not normalize the image to [0,1]
        Note: Use the 'L' flag while converting using Pillow's function

        Args:
            path: the file path to where the image is located on disk

        Returns:
            img: grayscale image with values in [0,255] loaded using Pillow.
        """

        img = None
        #######################################################################
        # Student code begins
        #######################################################################

        raise NotImplementedError('`load_img_from_path` function in '
            + '`dl_utils.py` needs to be implemented')

        #######################################################################
        # Student code ends
        #######################################################################
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hints:
        1) Get info from self.dataset
        2) Ise load_img_from_path
        3) Apply transforms if valid

        Args:
            index: Index

        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None
        #######################################################################
        # Student code starts
        #######################################################################

        raise NotImplementedError('`__getitem__` function in '
            + '`dl_utils.py` needs to be implemented')

        #######################################################################
        # Student code ends
        #######################################################################
        return img, class_idx

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """

        l = 0

        #######################################################################
        # Student code starts
        #######################################################################

        raise NotImplementedError('`__len__` function in '
            + '`dl_utils.py` needs to be implemented')

        #######################################################################
        # Student code ends
        #######################################################################
        return l
