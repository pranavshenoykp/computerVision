import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNet class to define the layers and loss function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNet, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(10, 20, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3)
            )

        conv_out = int(20*5*5)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out, 100),
            nn.Linear(100, 15)
            )

        self.loss_criterion = nn.MSELoss(reduction='mean')

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None
        #######################################################################
        # Student code begins
        #######################################################################

        (N,C,H,W) = x.shape

        conv_features = self.conv_layers(x)
  
        flat_features = conv_features.reshape(-1, 500)
        model_output = self.fc_layers(flat_features)


        #######################################################################
        # Student code ends
        #######################################################################
        return model_output
