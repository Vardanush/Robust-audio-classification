import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch import Tensor

__all__ = ["log_amp_mel_spectrogram", "FixedLength"]


def log_amp_mel_spectrogram(cfg=None):
    """
    Using original sampling_rate, n_fft, hop_length, n_mels from the CRNN paper
    """
    return nn.Sequential(
        T.MelSpectrogram(sample_rate=12000, n_fft=512, hop_length=256, n_mels=96),
        T.AmplitudeToDB()
    )