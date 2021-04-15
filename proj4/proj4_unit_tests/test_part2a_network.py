"""
Tests for the network
"""
import torch
from functools import reduce
import os

from proj4_code.part2a_network import MCNET
from proj4_code.part2b_patch import gen_patch


def test_mcnet():
    mcnet = MCNET(ws=11, batch_size=1, use_cuda=torch.cuda.is_available())
    assert mcnet(torch.Tensor(2, 1, 11, 11)).shape == torch.Size([1, 1])