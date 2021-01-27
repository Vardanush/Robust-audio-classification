'''
Adapted from: https://github.com/Hadisalman/smoothing-adversarial
Adapted from project 2, course: machine learning for graphs and sequential data
Paper: https://github.com/Hadisalman/smoothing-adversarial
'''
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
from .attacks import Attacker, PGD_L2

__all__ = ['SmoothADV']

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

class SmoothADV(Classifier, ABC):
    """
    Randomized smoothing classifier with adversarial training
    """
    # to abstain, Smooth returns this int
    ABSTAIN = -1
    
    def __init__(self, cfg, class_weights, base_classifier: Classifier, device, trial_hparams = None, train_loader = None, val_loader = None):
        """
        Constructor for SmoothADV.
        Parameters
        ----------
        base_classifier: pl.LightningModule
            The base classifier (i.e. f(x)) that maps an input sample to a logit vector.
        num_classes: int
            The number of classes.
        sigma: float
            The variance used for the Gaussian perturbations.
        """
        super(SmoothADV, self).__init__(class_weights, cfg["MODEL"]["NUM_CLASSES"], trial_hparams, train_loader, val_loader)
        
        self.save_hyperparameters(cfg)
        self.learning_rate = cfg["SOLVER"]["LEARNING_RATE"]
        self.weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"]
        self.step_size = cfg["SOLVER"]["STEP_SIZE"]
        self.gamma = cfg["SOLVER"]["GAMMA"]
        self.include_top = cfg["MODEL"]["CRNN"]["INCLUDE_TOP"]
        self.include_transform = cfg["MODEL"]["CRNN"]["INCLUDE_TRANSFORM"]
        
        self.base_classifier = base_classifier
        self.this_device = device
        self.num_classes = cfg["MODEL"]["NUM_CLASSES"]
        self.sigma = cfg["SOLVER"]["SIGMA"]
        self.attack = True if cfg["ATTACK"] else False
        self.epsilon = cfg["ATTACK_VAL"]["EPS"]
        self.num_steps = cfg["ATTACK_VAL"]["NUM_STEPS"]
        self.attacker = PGD_L2(steps=self.num_steps, device=device, max_norm=self.epsilon)
        self.mtrain = cfg["ATTACK_VAL"]["MTRAIN"]
        self.no_grad = cfg["ATTACK_VAL"]["NO_GRAD"]
        self.multi_noise = cfg["ATTACK_VAL"]["MULTI_NOISE"]
        self.val_mtrain = 2*cfg["ATTACK_VAL"]["MTRAIN"]

    def forward(self, x, seq_len):
        """
        Make a single prediction for the input batch using the base classifier

        Parameters
        ----------
        inputs
        
        Returns
        -------
        torch.Tensor of shape [B, K] where K is the number of classes
        """
        return self.base_classifier(x, seq_len) 
    
    """
    Added from CRNN
    """
    def training_step(self, batch, batch_idx):
        mini_batches = self.get_minibatches(batch, self.mtrain)
        self.noise_list = []
        main_loss = []
        main_acc = []
        
        for x, y, original_lengths in mini_batches:

            """
            Randomised smoothing
            """
            x = x.repeat((1, self.mtrain, 1, 1)).view(batch[0].shape)
            original_lengths = original_lengths.repeat(self.mtrain)
            noise = torch.randn_like(x, device=self.this_device) * self.sigma

            """
            Adversarial training
            """
            for param in self.parameters(): 
                param.requires_grad_(False)
                
            self.eval()
            x = self.attacker.attack(self, x, y, original_lengths, 
                                            noise=noise, 
                                            num_noise_vectors=self.mtrain, 
                                            no_grad=self.no_grad
                                            )
            
            self.train()
            for param in self.parameters(): 
                param.requires_grad_(True)
            
            if self.multi_noise:
                x = x + noise
                y = y.unsqueeze(1).repeat(1, self.mtrain).reshape(-1,1).squeeze()
                out = self(x, original_lengths)
                    
                if self.class_weights is not None:
                    loss = F.cross_entropy(out, y, weight=self.class_weights)
                else:
                    loss = F.cross_entropy(out, y)
                    
                preds = torch.argmax(out, dim=1)
                acc = accuracy(preds, y)
                main_loss.append(loss)
                main_acc.append(acc)
            else:
                x = x[::self.mtrain] # subsample the samples
                noise = noise[::self.mtrain]
                self.noise_list.append(x + noise)
                
        if not self.multi_noise:
            x = torch.cat(self.noise_list)
            y = batch[1]
            assert len(y) == len(x)

            out = self(x, original_lengths)
            if self.class_weights is not None:
                loss = F.cross_entropy(out, y, weight=self.class_weights)
            else:
                loss = F.cross_entropy(out, y)
                    
            preds = torch.argmax(out, dim=1)
            acc = accuracy(preds, y)
        else:
            loss = sum(main_loss)/len(main_loss)
            acc = sum(main_acc)/len(main_acc)

        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, original_lengths = batch
        
        noise = torch.randn_like(x, device=self.this_device) * self.sigma
                    
        acc_normal = accuracy(torch.argmax(self(x + noise, original_lengths), dim=1), y)
        '''
        with torch.enable_grad():

            x = self.attacker.attack(self, x, y, original_lengths, noise=noise)
      
        x = x + noise
        out = self(x, original_lengths)

        if self.class_weights is not None:
            loss = F.cross_entropy(out, y, weight=self.class_weights)
        else:
            loss = F.cross_entropy(out, y)

        preds = torch.argmax(out, dim=1)
        acc = accuracy(preds, y)
        '''
        mini_batches = self.get_minibatches(batch, self.val_mtrain)
        self.noise_list = []
        main_loss = []
        main_acc = []
        main_prec = []
        main_recall = []
        
        for x, y, original_lengths in mini_batches:

            """
            Randomised smoothing
            """
            '''
            print("x shape in validation", x.shape)
            print("batch[0] shape", batch[0].shape)
            print("batch shape", len(batch))
            print("mtrain", self.mtrain)
            print("x repeat", x.repeat((1, self.mtrain, 1, 1)).shape)
            '''
            x = x.repeat((1, self.val_mtrain, 1, 1)).view(batch[0].shape)
            original_lengths = original_lengths.repeat(self.val_mtrain)
            noise = torch.randn_like(x, device=self.this_device) * self.sigma

            """
            Adversarial training
            """
            x = self.attacker.attack(self, x, y, original_lengths, 
                                            noise=noise, 
                                            num_noise_vectors= self.val_mtrain, 
                                            no_grad=self.no_grad
                                            )
            
            if self.multi_noise:
                x = x + noise
                y = y.unsqueeze(1).repeat(1, self.val_mtrain).reshape(-1,1).squeeze()
                out = self(x, original_lengths)
                    
                if self.class_weights is not None:
                    loss = F.cross_entropy(out, y, weight=self.class_weights)
                else:
                    loss = F.cross_entropy(out, y)
                    
                preds = torch.argmax(out, dim=1)
                acc = accuracy(preds, y)
                precision = self.val_precision(preds, y)
                recall = self.val_recall(preds, y)
                
                main_loss.append(loss)
                main_acc.append(acc)
                main_prec.append(precision)
                main_recall.append(recall)
            else:
                x = x[::self.val_mtrain] # subsample the samples
                noise = noise[::self.val_mtrain]
                self.noise_list.append(x + noise)
            
                
        if not self.multi_noise:
            x = torch.cat(self.noise_list)
            y = batch[1]
            assert len(y) == len(x)

            out = self(x, original_lengths)
            if self.class_weights is not None:
                loss = F.cross_entropy(out, y, weight=self.class_weights)
            else:
                loss = F.cross_entropy(out, y)
                    
            preds = torch.argmax(out, dim=1)
            acc = accuracy(preds, y)
        else:
            loss = sum(main_loss)/len(main_loss)
            acc = sum(main_acc)/len(main_acc)
            precision = sum(main_prec)/len(main_prec)
            recall = sum(main_recall)/len(main_recall)
        
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_acc_normal', acc_normal, on_epoch=True, prog_bar=True)
        self.log('val_precision', precision, prog_bar=False)
        self.log('val_recall', recall, prog_bar=False)
        
        return loss, y, preds
 
    
    def test_step(self, batch, batch_idx):
        x, y, original_lengths = batch
        
        noise = torch.randn_like(x, device=self.this_device) * self.sigma
        
        with torch.enable_grad():
            x = self.attacker.attack(self, x, y, original_lengths, noise=noise)
      
        x = x + noise
        
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
    
    def get_minibatches(self, batch, num_batches):
        X = batch[0]
        y = batch[1]
        seq_len = batch[2]

        batch_size = len(X) // num_batches

        for i in range(num_batches):
            if i*batch_size <len(X):
                
                yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size], seq_len[i*batch_size : (i+1)*batch_size]

                    
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
            return SmoothADV.ABSTAIN, 0.0
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
            return SmoothADV.ABSTAIN
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
            classes = torch.arange(self.num_classes).cuda()
         #   classes = torch.arange(self.num_classes)
            class_counts = torch.zeros([self.num_classes], dtype=torch.long).cuda()
 #           class_counts = torch.zeros([self.num_classes], dtype=torch.long)
            for it in range(ceil(num_samples / batch_size)):
                this_batch_size = min(num_remaining, batch_size)
                if self.include_transform:
                    batch = inputs.repeat((this_batch_size, 1, 1)) # if inputs are audios
                else:
                    batch = inputs.repeat((this_batch_size, 1, 1, 1)) # if inputs are melspectrogram
                random_noise = torch.randn_like(batch).cuda() * torch.tensor(self.sigma).cuda() # add random noise here
 #               random_noise = torch.randn_like(batch)* torch.tensor(self.sigma) # add random noise here
                seq_lens = seq_len.repeat(this_batch_size)
                
                predictions = self.base_classifier((batch + random_noise), seq_lens)
                class_counts += (predictions.argmax(-1, keepdim=True) == classes).long().sum(0)

        return class_counts



