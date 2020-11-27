from abc import ABC

import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

__all__ = ['Classifier']

class Classifier(pl.LightningModule, ABC):
    """
    Abstract base class for classifier models.
    """

    def __init__(self, class_weights, trial_hparams, train_loader, val_loader):
        super().__init__()
        self.class_weights = class_weights
        self.hparameters = trial_hparams
        self.train_loader = train_loader
        self.val_loader = val_loader

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self(x)
        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self(x)
        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self(x)
        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    
    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        
        if self.hparameters is not None:
            scheduler = None
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.hparameters["learning_rate"],
                weight_decay=self.hparameters["weight_decay"])
        else:
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)
            """
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, self.step_size,
                gamma=self.gamma)
            """
              
        return [optimizer]
