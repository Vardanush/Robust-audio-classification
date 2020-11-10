"""
Very Deep CNN models
Implementation from paper https://arxiv.org/pdf/1610.00087.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

__all__ = ["DeepCNN", "m11", "m18", "LitDeepCNN", "lit_m11", "lit_m18"]

def init_weights(m):
    if type(m) == nn.Conv1d:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
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
        
        self.model.apply(init_weights) # glorot initialisation of conv layers
        nn.init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x

class LitDeepCNN(pl.LightningModule):
    def __init__(self, num_classes, num_layers, learning_rate=0.001, 
                 weight_decay=1e-7, lr_scheduler_factor=0.5, lr_scheduler_patience=5, #1e-4 from the paper
                 lr_scheduler_threshold=0.001, lr_scheduler_monitor='val_acc'):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                  'lr_scheduler_factor', 'lr_scheduler_patience',
                                  'lr_scheduler_threshold', 'lr_scheduler_monitor')

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
        
        self.model.apply(init_weights) # glorot initialisation of conv layers
        nn.init.xavier_uniform(self.linear.weight)
        self.linear.bias.data.fill_(0.01)
        
    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=2)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.hparams.lr_scheduler_factor,
                patience=self.hparams.lr_scheduler_patience,
                threshold=self.hparams.lr_scheduler_threshold, verbose=True),
            'monitor': self.hparams.lr_scheduler_monitor,
        }
        return [optimizer], [scheduler]


def m11(num_classes):
    model = DeepCNN(num_classes, [2, 2, 3, 2])
    return model

def m18(num_classes):
    model = DeepCNN(num_classes, [4, 4, 4, 4])
    return model

def lit_m11(num_classes):
    model = LitDeepCNN(num_classes, [2, 2, 3, 2])
    return model

def lit_m18(num_classes):
    model = LitDeepCNN(num_classes, [4, 4, 4, 4])
    return model
