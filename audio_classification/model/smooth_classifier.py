"""
Adapted from project 2, course: machine learning for graphs and sequential data
"""
from abc import ABC

import torch
import torch.nn.functional as F
from scipy.stats import norm, binom_test
from torch import nn
import torch.optim as optim
from statsmodels.stats.proportion import proportion_confint
from math import ceil
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix

from .classifier import Classifier

__all__ = ['SmoothClassifier']

def lower_confidence_bound(num_class_A: int, num_samples: int, alpha: float) -> float:
    """
    Computes a lower bound on the probability of the event occuring in a Bernoulli distribution.
    Parameters
    ----------
    num_class_A: int
        The number of times the event occured in the samples.
    num_samples: int
        The total number of samples from the bernoulli distribution.
    alpha: float
        The desired confidence level, e.g. 0.05.

    Returns
    -------
    lower_bound: float
        The lower bound on the probability of the event occuring in a Bernoulli distribution.

    """
    return proportion_confint(num_class_A, num_samples, alpha=2 * alpha, method="beta")[0]

class SmoothClassifier(Classifier, ABC):
    """
    Randomized smoothing classifier.
    """
    # to abstain, Smooth returns this int
    ABSTAIN = -1
    
    def __init__(self, cfg, class_weights, base_classifier: Classifier, trial_hparams = None, train_loader = None, val_loader = None):
        """
        Constructor for SmoothClassifier.
        Parameters
        ----------
        base_classifier: pl.LightningModule
            The base classifier (i.e. f(x)) that maps an input sample to a logit vector.
        num_classes: int
            The number of classes.
        sigma: float
            The variance used for the Gaussian perturbations.
        """
        super(SmoothClassifier, self).__init__(class_weights, cfg["MODEL"]["NUM_CLASSES"], trial_hparams, train_loader, val_loader)
        
        self.save_hyperparameters(cfg)
        self.learning_rate = cfg["SOLVER"]["LEARNING_RATE"]
        self.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        self.step_size = cfg["SOLVER"]["STEP_SIZE"]
        self.gamma = cfg["SOLVER"]["GAMMA"]
        self.include_top = cfg["MODEL"]["CRNN"]["INCLUDE_TOP"]
        self.include_transform = cfg["MODEL"]["CRNN"]["INCLUDE_TRANSFORM"]
        
        self.base_classifier = base_classifier
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]
        self.sigma = cfg["SOLVER"]["SIGMA"]
        self.attack = True if cfg["ATTACK"] else False

    def forward(self, x, seq_len):
        """
        Make a single prediction for the input batch using the base classifier and random Gaussian noise.

        Parameters
        ----------
        inputs
        
        Returns
        -------
        torch.Tensor of shape [B, K] where K is the number of classes
        """
#        noise = torch.zeros(x.shape).cuda() # makes sure that the padded 0s remain unchanged  
        noise = torch.zeros(x.shape).cpu()
              
        if not self.attack:
            for i in range(x.shape[0]): # for each sample in batch
                temp_noise = torch.randn_like(x[i][:seq_len.data[i]], dtype=torch.float32).cuda() * torch.tensor(self.sigma).cuda()
#                 temp_noise = torch.randn_like(x[i][:seq_len.data[i]], dtype=torch.float32) * torch.tensor(self.sigma)
                noise[i][:seq_len.data[i]] = temp_noise

        return self.base_classifier(x + noise, seq_len) 
    
    """
    Added from CRNN
    """
    def training_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        
        out = self(x, original_lengths)

        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        out = self(x, original_lengths)

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
        x, y, original_lengths = batch
        out = self(x, original_lengths)
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


    def certify(self, inputs: torch.Tensor, n0: int, num_samples: int, alpha: float, batch_size: int, seq_len:int):
        """
        Certify the input sample using randomized smoothing.

        Uses lower_confidence_bound to get a lower bound on p_A, the probability of the top class.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N x N is the audio dim.
            The input audio to certify.
        n0: int
            Number of samples to determine the most likely class.
        num_samples: int
            Number of samples to use for the robustness certification.
        alpha: float
            The confidence level, e.g. 0.05 for an expected error rate of 5%.
        batch_size: int
           The batch size to use during the certification, i.e. how many noise samples to classify in parallel.
        seq_len: int
            Length of the audio sequence

        Returns
        -------
        Tuple containing:
            * top_class: int. The predicted class g(x) of the input sample x. Returns -1 in case the classifier abstains
                         because the desired confidence level could not be reached.
            * radius: float. The radius for which the prediction can be certified. Is zero in case the classifier
                      abstains.

        """
        self.base_classifier.eval()

        class_counts_selection = self._sample_noise_predictions(inputs, n0, batch_size, seq_len)
        top_class = class_counts_selection.argmax().item()

        counts_estimation = self._sample_noise_predictions(inputs, num_samples, batch_size, seq_len)
        num_top_class = counts_estimation[top_class].item()
        p_A_lower_bound = lower_confidence_bound(num_top_class, num_samples, alpha)
                                    
        if p_A_lower_bound < 0.5:
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(p_A_lower_bound)
            return top_class, radius

    def predict(self, inputs: torch.tensor, num_samples: int, alpha: float, batch_size: int, seq_len: int) -> int:
        """
        Predict a label for the input sample via the smooth classifier g(x).

        Uses the test binom_test(count1, count1+count2, p=0.5) > alpha to determine whether the top class is the winning
        class with at least the confidence level alpha.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N x N is the audio dim.
            The input audio to certify.
        num_samples: int
            The number of samples to draw in order to determine the most likely class.
        alpha: float
            The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
            the expected error rate must not be larger than 5%.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise samples to classify in parallel.
        seq_len: int
            Length of the audio sequence

        Returns
        -------
        int: the winning class or -1 in case the desired confidence level could not be reached.
        """
        self.base_classifier.eval()
        class_counts = self._sample_noise_predictions(inputs, num_samples, batch_size, seq_len).cpu()
        top_2_classes = class_counts.argsort()[-2:]

        count1 = class_counts[top_2_classes[0]]
        count2 = class_counts[top_2_classes[1]]

        if binom_test(count1, count1+count2, p=0.5) > alpha:
            return SmoothClassifier.ABSTAIN
        else:
            return top_2_classes[1].item()

    def _sample_noise_predictions(self, inputs: torch.tensor, num_samples: int, batch_size: int, seq_len:int) -> torch.Tensor:
        """
        Sample random noise perturbations for the input sample and count the predicted classes of the base classifier.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N x N is the audio dim.
            The input audio to certify.
        num_samples: int
            The number of samples to draw.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise samples to classify in parallel.
        seq_len: int
            Length of the audio sequence

        Returns
        -------
        torch.Tensor of shape [K,], where K is the number of classes.
        Each entry of the tensor contains the number of times the base classifier predicted the corresponding class for
        the noise samples.
        """
        num_remaining = num_samples
        with torch.no_grad():
         #   classes = torch.arange(self.num_classes).cuda()
            classes = torch.arange(self.num_classes)
 #           class_counts = torch.zeros([self.num_classes], dtype=torch.long).cuda()
            class_counts = torch.zeros([self.num_classes], dtype=torch.long)
            for it in range(ceil(num_samples / batch_size)):
                this_batch_size = min(num_remaining, batch_size)
                if self.include_transform:
                    batch = inputs.repeat((this_batch_size, 1, 1)) # if inputs are audios
                else:
                    batch = inputs.repeat((this_batch_size, 1, 1, 1)) # if inputs are melspectrogram
#                random_noise = torch.randn_like(batch).cuda() * torch.tensor(self.sigma).cuda() # add random noise here
                random_noise = torch.randn_like(batch)* torch.tensor(self.sigma) # add random noise here
                seq_lens = seq_len.repeat(this_batch_size)
                
                predictions = self.base_classifier((batch + random_noise), seq_lens)
                class_counts += (predictions.argmax(-1, keepdim=True) == classes).long().sum(0)

        return class_counts

