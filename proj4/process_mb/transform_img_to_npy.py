"""
Transform imgs and disparity map from Middlebury dataset to .npy format
"""
#!/usr/bin/python3

import glob
import os
import re
import sys
import subprocess
import cv2

import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# two function to load the .pfm data
from preprocess_mb import load_pfm, save_pfm # load the .pfm data from original source code
# from readpfm import load_pfm # load the .pfm data by code of gengshan-y's repo


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


def read_pfm_by_mccnn(filename, scene):
    disp_path_left = filename + "/disp0.pfm"
    ## this load_pfm should use python2 to run, so first save the data returned by load_pfm to original_disp0
    ## and then use mask to further process this to get real disp, the load_pfm will downsample the disp
    disp, _ = load_pfm(disp_path_left, downsample=False)  # or try load_pfm
    disp = np.array(disp)
    np.save("/home/jingwu/git/Middlebury_data/" + scene + "_pfm_disp.npy", disp)


def transform_to_npy(filename, scene):
    """Transform datasets from .png/.pfm to .npy."""

    img0_path = filename + "/im0.png"
    img1_path = filename + "/im1.png"
    original_disp0 = "/home/jingwu/git/Middlebury_data/" + scene + "_pfm_disp.npy"

    # read and transform left img
    with open(img0_path, "rb") as f:
        img0 = Image.open(f)
        img0 = img0.convert("L")
        img0 = np.array(img0)
        # downsample
        img0 = img0[::2, ::2]

    # read and transform right img
    with open(img1_path, "rb") as f:
        img1 = Image.open(f)
        img1 = img1.convert("L")
        img1 = np.array(img1)
        # downsample
        img1 = img1[::2, ::2]

    read_pfm_by_mccnn(filename, scene)
    # read npy saved by load_pfm
    masked_disp = np.load(original_disp0)
    # subprocess.Popen("./computemask {} {} -1 {}/mask.png".format(disp_path_left, disp_path_right, filename).split())
    mask = Image.open("{}/mask.png".format(filename))
    mask = np.array(mask)[::2, ::2]
    masked_disp[mask != 255] = 0

    # # generate nnz
    # y, x = np.nonzero(mask == 255)
    # valid_disp = disp[y, x].reshape(-1)
    # nnz = np.vstack((y, x, valid_disp))
    # print("nnz from new dataset")

    assert img0.shape == masked_disp.shape
    assert img1.shape == masked_disp.shape
    np.save("/home/jingwu/git/Middlebury_data/" + scene + "_left.npy", img0)
    np.save("/home/jingwu/git/Middlebury_data/" + scene + "_right.npy", img1)
    np.save("/home/jingwu/git/Middlebury_data/" + scene + "_disp.npy", masked_disp)


data_dir = "path/to/mb_data"
ALL_SCENES = TEST_SCENES + TRAIN_SCENES + VAL_SCENES
for scene in ALL_SCENES:
    print(scene)
    scene_path = data_dir + scene
    scene_path += "-imperfect"
    transform_to_npy(scene_path, scene)
    print("*****finished*****")
