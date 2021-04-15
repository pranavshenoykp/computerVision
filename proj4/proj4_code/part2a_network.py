#!/usr/bin/python3

"""Defines network architectures."""

import os

import torch
from torch import nn


def save_model(network, path) -> None:
    torch.save(network.state_dict(), path)


def load_model(
    network: nn.Module, path: str, device: str = "cpu", strict: bool = True
) -> nn.Module:
    network.load_state_dict(
        torch.load(path, map_location=torch.device(device)), strict=strict
    )

    return network


class MCNET(torch.nn.Module):
    """MCNET based on paper from [Zbontar & LeCun, 2015].

    This network takes as input two patches of size ws x ws and outputs the
    likelihood of the two patches being a match.
    """

    def __init__(
        self,
        ws=11,
        batch_size: int = 512,
        load_path: str = None,
        strict: bool = True,
        use_cuda: bool = True,
    ) -> None:
        """
        MCNET class constructor.

        Args:
            ws: window size (or blocking size) of the input patch
            batch_size: number of patch pairs inside each batch
        """
        super(MCNET, self).__init__()

        num_feature_map = 112
        kernel_size = 3
        num_hidden_unit = 384
        self.batch_size = batch_size
        self.ws = ws
        self.strict = strict
        device = torch.device("cuda" if use_cuda else "cpu")

        self.conv = None  # placeholder
        self.classifier = None  # placeholder
        #######################################################################
        # Student code begins
        #######################################################################

        raise NotImplementedError(
            "`self.conv` and `self.classifier` for MCNET in "
            + "`part2a_network.py` needs to be implemented"
        )

        #######################################################################
        # Student code ends
        #######################################################################

        self.criterion = nn.BCELoss().to(device)

        if load_path is not None and os.path.exists(load_path):
            self.load_state_dict(torch.load(load_path), strict=strict)
            self.to(device)

    def forward(self, x) -> torch.Tensor:
        """
        Calculates the matching cost between a batch of doubly-stacked paired
        tensors.

        Args:
            x: A tensor of shape (2 * batch_size, 1, ws, ws) batch of paired
                patches (e.g., the batch looks like
                [a_0, b_0, a_1, b_1, ... a_{batch_size}, b_{batch_size}], where
                a, b are (1, ws, ws) patches)
        Returns:
            output: Tensor of shape (batch_size,1) representing the matching
                cost computed by MC-CNN over the batch of paired patches we use
                0 for a positive match (represent 0 cost to match) and 1 for a
                negative match
        """
        conv_features = self.conv(x)
        flat_features = conv_features.reshape(self.batch_size, -1)
        out = self.classifier(flat_features)
        return out
