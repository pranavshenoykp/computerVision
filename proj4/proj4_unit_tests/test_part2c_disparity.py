import torch
from torch import nn
import numpy as np

from proj4_code.part2c_disparity import (
    calculate_mccnn_disparity_map,
    mc_cnn_similarity,
    calculate_mccnn_cost_volume,
)
from proj4_code.utils import generate_delta_fn_images


class FakeConv(nn.Module):
    """Loops over the channels till the number of output channels is reached

    Args:
    -   in_channels: number of input channels
    -   out_channels: number of input channels
    """

    def __init__(self, in_channels, out_channels):
        super(FakeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return torch.cat(
            [
                x[:, i % self.in_channels, :, :].unsqueeze(1)
                for i in range(self.out_channels)
            ],
            dim=1,
        )


class FakeClassifier(nn.Module):
    """Computes the similarity between the inputs by taking the sum of their pairwise
    product as an arbitrary measure to test whether patches are being matched correctly.

    Args:
    -   in_features: number of input channels
    Returns:
    -   A stable (but arbitrary) similarity measure for each item in the batch
    """

    def __init__(self, in_features):
        super(FakeClassifier, self).__init__()
        self.in_features = in_features

    def forward(self, x):
        batch_size, num_features = x.shape
        half_size = num_features // 2
        left_half = x[:, :half_size]
        right_half = x[:, half_size:]
        assert (
            left_half.shape == right_half.shape
        ), "Number of features sent to classifier must be even"
        return torch.cat(
            [
                torch.sum((left_half[i, ...] - right_half[i, ...])).unsqueeze(0)
                for i in range(batch_size)
            ],
            dim=0,
        )


class ToyMCNET(nn.Module):
    """Toy MCNET-like network to test disparity computation

    Calculates an arbitrary measure instead of the MCNET measure

    Args:
    -   ws: window size (or blocking size) of the input patch
    -   batch_size: batch size
    -   channels: the number of input and output channels for the conv layers
    Returns:
    -   matching cost between the 2 patches, we use 0 for positive match
        (represent 0 cost to match) and 1 for negative match
    """

    def __init__(self, ws=11, batch_size=512, num_channels=1):
        super(ToyMCNET, self).__init__()

        self.batch_size = batch_size
        self.ws = ws
        device = torch.device("cpu")

        # This is NOT the expected structure of the MCNET from part2a
        # Copying from this is futile
        self.conv = nn.Sequential(
            FakeConv(in_channels=num_channels, out_channels=num_channels),
        ).to(device)

        self.classifier = nn.Sequential(FakeClassifier(in_features=(ws ** 2) * 2)).to(
            device
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(self.batch_size, -1)
        return self.classifier(x)


def test_disparity_mccnn_deltafn_success():
    """
    Tests the disparity map giving inputs which just have a single pixel value
    """
    im_dim = 51
    block_size = 1

    # Generates 2 images with just 1 pixel difference
    im_left, im_right = generate_delta_fn_images((im_dim, im_dim))
    im_left = im_left.transpose(0, 2).transpose(1, 2)
    im_right = im_right.transpose(0, 2).transpose(1, 2)

    # calculate the disparity manually
    left_idx = torch.argmax(im_left[:, :, 1]).item()
    left_r = left_idx // im_dim
    left_c = left_idx - left_r * im_dim

    right_idx = torch.argmax(im_right[:, :, 1]).item()
    right_r = right_idx // im_dim
    right_c = right_idx - right_r * im_dim

    disparity_expected = right_c - left_c

    net = ToyMCNET(ws=block_size, batch_size=1, num_channels=3)

    # get the disparity map from the function
    disp_map = calculate_mccnn_disparity_map(
        net,
        im_left,
        im_right,
        block_size,
        mc_cnn_similarity,
        max_search_bound=disparity_expected + 3,
    )

    # we should get two non-zero values in the disparity map
    nonzero_disp = torch.nonzero(disp_map).data

    # check the size
    assert nonzero_disp.size() == (2, 2)

    # check that the rows are same
    assert nonzero_disp[0, 0].item() == nonzero_disp[1, 0].item()

    val1 = disp_map[left_r, left_c].item()

    # Check that the disparity is in the expected location
    assert val1 == disparity_expected


def test_calculate_mccnn_disparity_map():
    """Tests the disparity map calculation using
    a contrived network that calculates SSD
    """
    window_size = 1
    mcnet = ToyMCNET(ws=window_size, batch_size=1, num_channels=1)

    toy_img_left = torch.Tensor(
        [
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    toy_img_right = torch.Tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    student_disparity = calculate_mccnn_disparity_map(
        mcnet,
        toy_img_left,
        toy_img_right,
        block_size=window_size,
        sim_measure_function=mc_cnn_similarity,
    )

    assert student_disparity.shape == torch.Size((7, 7))

    # Ground truth disparity cost map
    # The disparity cost is 0 wherever the
    # new positions of the pixels are as the image is sparse.
    # The cost differs from left-to-right here
    # since we don't use an abolute difference.
    ground_truth_disparity = torch.Tensor(
        [
            [0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 1, 2, 3, 4],
            [0, 0, 0, 1, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [0, 0, 0, 1, 2, 3, 4],
        ]
    ).long()
    assert torch.allclose(
        ground_truth_disparity, student_disparity
    ), "Disparity mismatch"


def test_calculate_mccnn_cost_volume():
    """Test calculate cost volume with simple dot"""
    left_image = torch.zeros((3, 10, 10))
    left_image[:, 8, 6] = 1
    right_image = torch.zeros((3, 10, 10))
    right_image[:, 8, 5] = 1

    window_size = 1

    mcnet = ToyMCNET(ws=window_size, batch_size=1, num_channels=1)

    cost_volume = calculate_mccnn_cost_volume(
        mcnet,
        left_image,
        right_image,
        block_size=1,
        sim_measure_function=mc_cnn_similarity,
        max_search_bound=4,
    )
    # minimum cost is at 1 because of 1px difference
    assert np.allclose(cost_volume[8, 6, :].cpu().numpy(), [1, 0, 1, 1])

    left_image = torch.zeros(((3, 10, 10)))
    left_image[:, 8, 7] = 1
    right_image = torch.zeros(((3, 10, 10)))
    right_image[:, 8, 1] = 1

    cost_volume = calculate_mccnn_cost_volume(
        mcnet,
        left_image,
        right_image,
        block_size=1,
        sim_measure_function=mc_cnn_similarity,
        max_search_bound=7,
    )

    # minimum cost is at 6 because of 6px difference
    assert np.all(np.isclose(cost_volume[8, 7, :].cpu().numpy(), [1, 1, 1, 1, 1, 1, 0]))


def test_calculate_mccnn_cost_volume_argmin():
    """Check whether the argmin of the cost volume is in the correct location"""
    window_size = 1

    mcnet = ToyMCNET(ws=window_size, batch_size=1, num_channels=1)

    left_image = torch.zeros(((3, 10, 10)))
    left_image[:, 5, 6] = 1
    right_image = torch.zeros(((3, 10, 10)))
    right_image[:, 5, 3] = 1

    cost_volume = calculate_mccnn_cost_volume(
        mcnet,
        left_image,
        right_image,
        block_size=window_size,
        sim_measure_function=mc_cnn_similarity,
        max_search_bound=7,
    )

    # Disparity value is 3 because of 3px difference
    assert np.argmin(cost_volume[5, 6, :].cpu().numpy()) == 3
