"""
Very Deep CNN models
Implementation from paper https://arxiv.org/pdf/1610.00087.pdf
"""

import torch
import torch.nn as nn

__all__ = ["DeepCNN", "m11", "m18"]


class DeepCNN(nn.Module):
    def __init__(self, num_classes, num_layers):
        super().__init__()

        modules = []

        modules.append(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4, padding=39))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(4))  # todo: padding = 1?

        for i in range(num_layers[0]):
            modules.append(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm1d(64))
            modules.append(nn.ReLU())

        modules.append(nn.MaxPool1d(4))
        modules.append(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())

        for i in range(num_layers[1]):
            modules.append(nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm1d(128))
            modules.append(nn.ReLU())

        modules.append(nn.MaxPool1d(4))
        modules.append(nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU())

        for i in range(num_layers[2]):
            modules.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm1d(256))
            modules.append(nn.ReLU())

        modules.append(nn.MaxPool1d(4))
        modules.append(nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())

        for i in range(num_layers[3]):
            modules.append(nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
            modules.append(nn.BatchNorm1d(512))
            modules.append(nn.ReLU())

        self.model = nn.Sequential(*modules)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x


def m11(num_classes):
    model = DeepCNN(num_classes, [2, 2, 3, 2])
    return model


def m18(num_classes):
    model = DeepCNN(num_classes, [4, 4, 4, 4])
    return model
