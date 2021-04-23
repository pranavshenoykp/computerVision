"""
Testing for models
"""

from collections import Counter
from typing import Union

import torch.nn as nn

from proj5_code.my_resnet import MyResNet18
from proj5_code.simple_net import SimpleNet
from proj5_code.simple_net_final import SimpleNetFinal


def flatten_layers(layers):
    """
    Keep on flattening nn.Sequential objects
    """

    flattened_layers = list()

    recurse = False
    if isinstance(layers, nn.Linear):
        return layers
    for elem in layers:
        if type(elem) == nn.Sequential:
            recurse = True
            flattened_layers += list(elem.children())
        else:
            flattened_layers.append(elem)

    if recurse:
        return flatten_layers(flattened_layers)

    return flattened_layers


def extract_model_layers(model: Union[SimpleNet, SimpleNetFinal, MyResNet18]):
    # get the CNN sequential
    layers = flatten_layers(
        list(model.conv_layers.children())
        + (
            [model.fc_layers]
            if isinstance(model.fc_layers, nn.Linear)
            else list(model.fc_layers.children())
        )
    )

    # generate counts of different types of layers present in the model
    layers_type = [x.__class__.__name__ for x in layers]
    layers_count = Counter(layers_type)

    # get the total number of parameters which require grad and which do not require grad
    num_params_grad = 0
    num_params_nograd = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params_grad += param.numel()
        else:
            num_params_nograd += param.numel()
    return (
        layers,
        layers[-1].out_features,
        layers_count,
        num_params_grad,
        num_params_nograd,
    )


if __name__ == "__main__":
    model1 = SimpleNet()
    print(extract_model_layers(model1))

    model2 = SimpleNetFinal()
    print(extract_model_layers(model2))

    model3 = MyResNet18()
    print(extract_model_layers(model3))
