from abc import ABC

import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix

__all__ = ['SmoothClassifier']

class SmoothClassifier(pl.LightningModule, ABC):
    """
    Randomized smoothing classifier.
    """

    def __init__(self, base_classifier: pl.LightningModule, num_classes: int, sigma: float):
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
        super(SmoothClassifier, self).__init__()
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def forward(self, x):
        """
        Make a single prediction for the input batch using the base classifier and random Gaussian noise.

        Note: this function clamps the distorted samples in the valid range, i.e. [0,1].
        Parameters
        ----------
        inputs
        Returns
        -------
        torch.Tensor of shape [B, K] where K is the number of classes
        """
        noise = torch.randn_like(inputs) * self.sigma
        return self.base_classifier((inputs + noise).clamp(0, 1))

    def certify(self, inputs: torch.Tensor, n0: int, num_samples: int, alpha: float, batch_size: int) -> Tuple[int, float]:
        """
        Certify the input sample using randomized smoothing.

        Uses lower_confidence_bound to get a lower bound on p_A, the probability of the top class.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to certify.
        n0: int
            Number of samples to determine the most likely class.
        num_samples: int
            Number of samples to use for the robustness certification.
        alpha: float
            The confidence level, e.g. 0.05 for an expected error rate of 5%.
        batch_size: int
           The batch size to use during the certification, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        Tuple containing:
            * top_class: int. The predicted class g(x) of the input sample x. Returns -1 in case the classifier abstains
                         because the desired confidence level could not be reached.
            * radius: float. The radius for which the prediction can be certified. Is zero in case the classifier
                      abstains.

        """
        self.base_classifier.eval()

        class_counts_selection = self._sample_noise_predictions(inputs, n0, batch_size)
        top_class = class_counts_selection.argmax().item()

        counts_estimation = self._sample_noise_predictions(inputs, num_samples, batch_size)
        num_top_class = counts_estimation[top_class].item()
        p_A_lower_bound = lower_confidence_bound(num_top_class, num_samples, alpha)
        if p_A_lower_bound < 0.5:
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(p_A_lower_bound)
            return top_class, radius

    def predict(self, inputs: torch.tensor, num_samples: int, alpha: float, batch_size: int) -> int:
        """
        Predict a label for the input sample via the smooth classifier g(x).

        Uses the test binom_test(count1, count1+count2, p=0.5) > alpha to determine whether the top class is the winning
        class with at least the confidence level alpha.

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw in order to determine the most likely class.
        alpha: float
            The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
            the expected error rate must not be larger than 5%.
        batch_size: int
            The batch si ze to use during the prediction, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        int: the winning class or -1 in case the desired confidence level could not be reached.
        """
        self.base_classifier.eval()
        class_counts = self._sample_noise_predictions(inputs, num_samples, batch_size).cpu()
        top_2_classes = class_counts.argsort()[-2:]

        count1 = class_counts[top_2_classes[0]]
        count2 = class_counts[top_2_classes[1]]

        if binom_test(count1, count1+count2, p=0.5) > alpha:
            return SmoothClassifier.ABSTAIN
        else:
            return top_2_classes[0]

    def _sample_noise_predictions(self, inputs: torch.tensor, num_samples: int, batch_size: int) -> torch.Tensor:
        """
        Sample random noise perturbations for the input sample and count the predicted classes of the base classifier.

        Note: this function clamps the distorted samples in the valid range, i.e. [0,1].

        Parameters
        ----------
        inputs: torch.Tensor of shape [1, C, N, N], where C is the number of channels and N is the image width/height.
            The input image to predict.
        num_samples: int
            The number of samples to draw.
        batch_size: int
            The batch size to use during the prediction, i.e. how many noise samples to classify in parallel.

        Returns
        -------
        torch.Tensor of shape [K,], where K is the number of classes.
        Each entry of the tensor contains the number of times the base classifier predicted the corresponding class for
        the noise samples.
        """
        num_remaining = num_samples
        with torch.no_grad():
            classes = torch.arange(self.num_classes).to(self.device())
            class_counts = torch.zeros([self.num_classes], dtype=torch.long, device=self.device())
            for it in range(ceil(num_samples / batch_size)):
                this_batch_size = min(num_remaining, batch_size)

                batch = inputs.repeat((this_batch_size, 1, 1, 1))
                random_noise = torch.randn_like(batch) * self.sigma

                predictions = self.base_classifier((batch + random_noise).clamp(0, 1))

                class_counts += (predictions.argmax(-1, keepdim=True) == classes).long().sum(0)
        return class_counts

