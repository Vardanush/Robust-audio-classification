import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

__all__ = ["CRNN", "LitCRNN"]


class CRNN(nn.Module):
    """
    Covolutional recurrent neural network.
    Implementation from paper https://arxiv.org/abs/1609.04243
    """
    def __init__(self, include_top=True, num_classes=10, pad_input=15):
        super().__init__()
        self.include_top = include_top
        self.num_classes = num_classes
        self.pad_input = pad_input

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.1)

        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.1)

        # Conv block 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.1)

        # Conv block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(p=0.1)

        # GRU block 1, 2, output
        self.rnn = nn.GRU(128, 32, 2)
        self.dropout_rnn = nn.Dropout(p=0.3)
        if self.include_top:
            self.linear = nn.Linear(32, self.num_classes)

    def forward(self, x):
        out = F.pad(x, (self.pad_input, self.pad_input))

        out = F.max_pool2d(F.elu(self.bn1(self.conv1(out))), 2, stride=2)
        out = self.dropout1(out)
        out = F.max_pool2d(F.elu(self.bn2(self.conv2(out))), 3, stride=3)
        out = self.dropout2(out)
        out = F.max_pool2d(F.elu(self.bn3(self.conv3(out))), 4, stride=4)
        out = self.dropout3(out)
        out = F.max_pool2d(F.elu(self.bn4(self.conv4(out))), 4, stride=4)
        out = self.dropout4(out)

        out = torch.squeeze(out, 2).permute(2, 0, 1)     # shape (N, H, 1, L) -> # (L, N ,H)
        _, out = self.rnn(out)   # get the final hidden_state
        out = self.dropout_rnn(out)
        out = out[-1]    # use only the hidden_state in the last RNN layer
        if self.include_top:
            out = self.linear(out)

        return out


class LitCRNN(pl.LightningModule):
    """
    Covolutional recurrent neural network. Pytorch-Lightning Version.
    Implementation from paper https://arxiv.org/abs/1609.04243
    """

    def __init__(self, include_top=True, num_classes=10, pad_input=15,
                 learning_rate=0.001, weight_decay=0.001,
                 lr_scheduler_factor=0.5, lr_scheduler_patience=5,
                 lr_scheduler_threshold=0.001, lr_scheduler_monitor='val_acc'):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay',
                                  'lr_scheduler_factor', 'lr_scheduler_patience',
                                  'lr_scheduler_threshold', 'lr_scheduler_monitor')

        self.include_top = include_top
        self.num_classes = num_classes
        self.pad_input = pad_input

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p=0.1)

        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(p=0.1)

        # Conv block 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.1)

        # Conv block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(p=0.1)

        # GRU block 1, 2, output
        self.rnn = nn.GRU(128, 32, 2)
        self.dropout_rnn = nn.Dropout(p=0.3)
        if self.include_top:
            self.linear = nn.Linear(32, self.num_classes)

    def forward(self, x):
        out = F.pad(x, (self.pad_input, self.pad_input))

        out = F.max_pool2d(F.elu(self.bn1(self.conv1(out))), 2, stride=2)
        out = self.dropout1(out)
        out = F.max_pool2d(F.elu(self.bn2(self.conv2(out))), 3, stride=3)
        out = self.dropout2(out)
        out = F.max_pool2d(F.elu(self.bn3(self.conv3(out))), 4, stride=4)
        out = self.dropout3(out)
        out = F.max_pool2d(F.elu(self.bn4(self.conv4(out))), 4, stride=4)
        out = self.dropout4(out)

        # shape (N, H, 1, L) -> # (L, N ,H)
        out = torch.squeeze(out, 2).permute(2, 0, 1)
        _, out = self.rnn(out)  # get the final hidden_state
        out = self.dropout_rnn(out)
        out = out[-1]  # use only the hidden_state in the last RNN layer
        if self.include_top:
            out = self.linear(out)

        return out

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