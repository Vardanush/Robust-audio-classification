import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from pytorch_lightning.metrics.functional import accuracy
from .classifier import Classifier

__all__ = ["LitCRNN"]


class LitCRNN(Classifier):
    """
    Covolutional recurrent neural network. Pytorch-Lightning Version.
    Implementation from paper https://arxiv.org/abs/1609.04243
    """

    def __init__(self, cfg, class_weights):
        super().__init__(class_weights)
        self.learning_rate = cfg["SOLVER"]["LEARNING_RATE"]
        self.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        self.step_size = cfg["SOLVER"]["STEP_SIZE"]
        self.gamma = cfg["SOLVER"]["GAMMA"]
        self.include_top = cfg["MODEL"]["CRNN"]["INCLUDE_TOP"]
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]

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

    def forward(self, x, seq_lens):
        out = F.max_pool2d(F.elu(self.bn1(self.conv1(x))), 2, stride=2)
        out = self.dropout1(out)
        out = F.max_pool2d(F.elu(self.bn2(self.conv2(out))), 3, stride=3)
        out = self.dropout2(out)
        out = F.max_pool2d(F.elu(self.bn3(self.conv3(out))), 4, stride=4)
        out = self.dropout3(out)
        out = F.max_pool2d(F.elu(self.bn4(self.conv4(out))), 4, stride=4)
        out = self.dropout4(out)

        # shape (N, H, 1, L) -> # (L, N ,H)
        out = torch.squeeze(out, 2).permute(2, 0, 1)
        seq_lens = torch.floor(seq_lens/96).type(torch.LongTensor)
        seq_lens = torch.where(seq_lens > 0, seq_lens, 1)
        # Use pack_padded_sequence to handle padded inputs
        out = pack_padded_sequence(out, seq_lens, enforce_sorted=False)

        _, out = self.rnn(out)  # get the final hidden_state
        out = self.dropout_rnn(out)
        out = out[-1]  # use only the hidden_state in the last RNN layer
        if self.include_top:
            out = self.linear(out)

        return out
    
    def training_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        out = self(x, original_lengths)
        loss = F.cross_entropy(out, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        out = self(x, original_lengths)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        out = self(x, original_lengths)
        loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
