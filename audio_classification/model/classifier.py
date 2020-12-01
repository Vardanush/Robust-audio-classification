from abc import ABC

import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Precision, Recall
from sklearn.metrics import classification_report

__all__ = ['Classifier']

class Classifier(pl.LightningModule, ABC):
    """
    Abstract base class for classifier models.
    """

    def __init__(self, class_weights, trial_hparams, train_loader, val_loader, num_classes):
        super(Classifier, self).__init__()
        self.class_weights = class_weights
        self.hparameters = trial_hparams
        self.train_loader = train_loader
        self.val_loader = val_loader
        """
         "macro" => computes precision and recall per class and takes the mean
         "micro" => computes precision and recall globally
        """
        self.val_precision = Precision(num_classes=num_classes, average='macro')
        self.val_recall = Recall(num_classes=num_classes, average='macro')
        self.test_precision = Precision(num_classes=num_classes, average='macro')
        self.test_recall = Recall(num_classes=num_classes, average='macro')
        self.num_classes = num_classes

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
        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        return loss, y, preds

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        out = self(x)
        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)
        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)
        precision = self.test_precision(preds, y)
        recall = self.test_recall(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        return loss, y, preds

    def train_dataloader(self):
        return self.train_loader

    """
    def validation_epoch_end(self, val_outputs):
        #todo; add precision and recall per class
        avg_loss = 0
        for output in val_outputs:
            true_label = output[1].cpu().numpy()
            predict =output[2].cpu().numpy()
            avg_loss +=output[0].cpu().numpy()
            report = classification_report(true_label,predict)
            print(report)
        #report = report/len(val_outputs)
        #print("loss on complete val set", avg_loss/len(val_outputs))
        #print(report)
        #return report
    """

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):

        if self.hparameters is not None:
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.hparameters["learning_rate"],
                weight_decay=self.hparameters["weight_decay"])

            return [optimizer]
        else:
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.StepLR(
                optimizer, self.step_size,
                gamma=self.gamma)

            return [optimizer], [scheduler]
