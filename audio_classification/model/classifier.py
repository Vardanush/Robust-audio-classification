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
        self.trial_hparams = trial_hparams
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

    def configure_optimizers(self):
        print(self.trial_hparams["learning_rate"])
        optimizer = optim.Adam(
                self.parameters(),
                lr=self.trial_hparams["learning_rate"],
                weight_decay=self.trial_hparams["weight_decay"])
        
        if self.trial_hparams is not None:
            scheduler = None
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, self.step_size,
                gamma=self.gamma)
            
        return [optimizer], [scheduler]
