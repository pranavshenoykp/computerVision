"""
Tests for the network
"""
import torch
from functools import reduce
import os

from proj4_code.part2b_patch import gen_patch


def test_gen_patch():
    image = torch.tensor(
        [
            [
                [156, 219, 57, 188, 105],
                [198, 148, 54, 74, 236],
                [196, 3, 147, 68, 81],
                [85, 132, 227, 27, 225],
                [250, 28, 168, 118, 12],
            ],
            [
                [83, 31, 221, 247, 236],
                [102, 229, 9, 221, 179],
                [49, 241, 223, 234, 48],
                [40, 167, 173, 98, 246],
                [93, 165, 229, 144, 96],
            ],
        ]
    )
    out = gen_patch(image, x=2, y=2, ws=3)

    # (2,3,3) patch with top-left corner at (2,2)
    gt = torch.tensor(
        [
            [[147.0, 68.0, 81.0], [227.0, 27.0, 225.0], [168.0, 118.0, 12.0]],
            [[223.0, 234.0, 48.0], [173.0, 98.0, 246.0], [229.0, 144.0, 96.0]],
        ]
    )
    assert torch.all(torch.isclose(out, gt))

    # test corner case
    out = gen_patch(image, x=4, y=4, ws=3)

    # (2,3,3) patch with top-left corner at (2,2)
    gt = torch.tensor(
        [
            [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[96.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    assert torch.all(torch.isclose(out, gt))
