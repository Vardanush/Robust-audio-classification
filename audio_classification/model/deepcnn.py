"""
Very Deep CNNs (M11, M18).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .classifier import Classifier

__all__ = ["LitDeepCNN", "lit_m11", "lit_m18"]


def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class LitDeepCNN(Classifier):
    """
    Very Deep CNN models. Pytorch-Lightning Version.
    Implementation from paper https://arxiv.org/pdf/1610.00087.pdf
    """

    def __init__(self, cfg, num_layers, class_weights, trial_hparams, train_loader, val_loader):
        super(LitDeepCNN, self).__init__(class_weights, cfg["MODEL"]["NUM_CLASSES"], trial_hparams, train_loader, val_loader)
        self.learning_rate = cfg["SOLVER"]["LEARNING_RATE"]
        self.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        self.step_size = cfg["SOLVER"]["STEP_SIZE"]
        self.gamma = cfg["SOLVER"]["GAMMA"]

        modules = []

        modules.append(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=80, stride=4, padding=39))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(4)) 

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
        self.linear = nn.Linear(512, self.num_classes)

        self.model.apply(init_weights)  # glorot initialisation of conv layers
        nn.init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x


class lit_m11(LitDeepCNN):

    def __init__(self, cfg, class_weights, num_layers=[2, 2, 3, 2], trial_hparams = None, train_loader = None, val_loader = None):
        LitDeepCNN.__init__(self, cfg, num_layers, class_weights, trial_hparams, train_loader, val_loader)


class lit_m18(LitDeepCNN):

    def __init__(self, cfg, class_weights, num_layers=[4, 4, 4, 4], trial_hparams = None, train_loader = None, val_loader = None):
        LitDeepCNN.__init__(self, cfg, num_layers, class_weights, trial_hparams, train_loader, val_loader)
