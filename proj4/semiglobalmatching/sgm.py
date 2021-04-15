"""
This code has been adapted from https://github.com/beaupreda/semi-global-matching/blob/master/sgm.py

python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

original author: David-Alexandre Beaupre
date: 2019/07/12
"""

from typing import Any, List, Tuple, Callable

import argparse
import sys
import time as t
import torch

import numpy as np

from proj4_code.part1c_disparity_map import calculate_cost_volume


class Direction:
    def __init__(self, direction: Tuple[int, int] = (0, 0), name: str = "invalid"):
        """Represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        Args:
            direction: (x, y) for cardinal direction.
            name: common name of said direction.
        """
        self.direction = direction
        self.name = name


# 8 defined directions for sgm
N = Direction(direction=(0, -1), name="north")
NE = Direction(direction=(1, -1), name="north-east")
E = Direction(direction=(1, 0), name="east")
SE = Direction(direction=(1, 1), name="south-east")
S = Direction(direction=(0, 1), name="south")
SW = Direction(direction=(-1, 1), name="south-west")
W = Direction(direction=(-1, 0), name="west")
NW = Direction(direction=(-1, -1), name="north-west")


class Paths:
    def __init__(self):
        """Represent the relation between the directions."""
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E, W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(
        self,
        max_search_bound: int = 64,
        P1: int = 5,
        P2: int = 70,
        csize: Tuple[int, int] = (7, 7),
        bsize: Tuple[int, int] = (3, 3),
    ):
        """Represent all parameters used in the sgm algorithm.
        Args:
            max_search_bound: maximum search distance between the same pixel in both images.
            P1: penalty for disparity difference = 1
            P2: penalty for disparity difference > 1
            csize: size of the kernel for the census transform.
            bsize: size of the kernel for blurring the images and median filtering.
        """
        self.max_search_bound = max_search_bound
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize


def get_indices(offset: int, dim: int, direction: int, height: int) -> np.ndarray:
    """For the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    Args:
            offset: difference with the main diagonal of the cost volume.
            dim: number of elements along the path.
            direction: current aggregation direction.
            height: H of the cost volume.
    Returns:
            arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice: np.ndarray, offset: int, parameters: Parameters) -> np.ndarray:
    """Part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    Args:
            slice: M x D array from the cost volume.
    Args:
            parameters: structure containing parameters of the algorithm.
    Returns:
            M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=np.float32)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=np.float32)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(
            disparity_dim, disparity_dim
        )
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(
    cost_volume: np.ndarray, parameters: Parameters, paths: Paths
) -> np.ndarray:
    """Second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    Args:
            cost_volume: array containing the matching costs.
            parameters: structure containing parameters of the algorithm.
            paths: structure containing all directions in which to aggregate costs.
    Returns:
            H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(
        shape=(height, width, disparities, paths.size), dtype=np.float32
    )

    path_id = 0
    for path in paths.effective_paths:
        print(
            "\tProcessing paths {} and {}...".format(path[0].name, path[1].name), end=""
        )
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(
            shape=(height, width, disparities), dtype=np.float32
        )
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(
                    get_path_cost(north, 1, parameters), axis=0
                )

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(
                    get_path_cost(west, 1, parameters), axis=0
                )

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(
                    south_east, 1, parameters
                )
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(
                    north_west, 1, parameters
                )

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(
                    south_west, 1, parameters
                )
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(
                    north_east, 1, parameters
                )

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print("\t(done in {:.2f} s)".format(dusk - dawn))

    return aggregation_volume


def compute_costs(
    left_img: np.ndarray,
    right_img: np.ndarray,
    max_search_bound: np.ndarray,
    sim_fn: Callable,
    block_size: int = 9,
    save_images: bool = False,
) -> np.ndarray:
    """First step of the sgm algorithm, matching cost based on census transform and hamming distance.
    Args:
            left: left image.
            right: right image.
            parameters: structure containing parameters of the algorithm.
            save_images: whether to save census images or not.
    Returns:
            H x W x D array with the matching costs.
    """
    use_cuda = True and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return (
        calculate_cost_volume(
            torch.FloatTensor(left_img),
            torch.FloatTensor(right_img),
            max_search_bound,
            sim_fn,
            block_size=block_size,
        )
        .cpu()
        .numpy()
        / block_size ** 2
    )


def select_disparity(aggregation_volume: np.ndarray) -> np.ndarray:
    """Last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    Args:
            aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    Returns:
            disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


def normalize(volume: np.ndarray, parameters: Parameters) -> np.ndarray:
    """Transforms values from the range (0, 64) to (0, 255).
    Args:
            volume: n dimension array to normalize.
            parameters: structure containing parameters of the algorithm.
    Returns:
            normalized array.
    """
    return 255.0 * volume / parameters.max_search_bound


def sgm(
    im_left: np.ndarray,
    im_right: np.ndarray,
    output_name: str,
    max_search_bound: int,
    sim_fn: Callable,
    block_size: int = 9,
    save_images: bool = False,
):
    """Main function applying the semi-global matching algorithm.
    Returns:
            disp_map: np.darray, the shape of disp_map is (H, w)
    """

    disparity = max_search_bound

    parameters = Parameters(
        max_search_bound=disparity,
        P1=8.0 / 255,
        P2=128.0 / 255,
        csize=(7, 7),
        bsize=(3, 3),
    )
    paths = Paths()

    left = im_left
    right = im_right

    print("\nStarting cost computation...")
    cost_volume = compute_costs(
        left, right, max_search_bound, sim_fn, block_size, save_images
    )

    print("\nStarting aggregation computation...")
    aggregation_volume = aggregate_costs(cost_volume, parameters, paths)

    print("\nSelecting best disparities...")
    disparity_map = np.float32(select_disparity(aggregation_volume))
    print("\nDone")
    return disparity_map


def sgm_mccnn(
    mccnn_cost_volume: np.ndarray,
    output_name: str,
    max_search_bound: int,
    sim_fn: Callable,
    block_size: int = 9,
    save_images: bool = False,
):
    """Main function applying the semi-global matching algorithm to a cost-volume generated by MC-CNN.
    Returns:
            disp_map: np.darray, the shape of disp_map is (H, w)
    """

    disparity = max_search_bound

    parameters = Parameters(
        max_search_bound=disparity,
        P1=8.0 / 255,
        P2=128.0 / 255,
        csize=(7, 7),
        bsize=(3, 3),
    )
    paths = Paths()

    print("\nStarting aggregation computation...")
    aggregation_volume = aggregate_costs(mccnn_cost_volume, parameters, paths)

    print("\nSelecting best disparities...")
    disparity_map = np.float32(select_disparity(aggregation_volume))
    print("\nDone")
    return disparity_map
