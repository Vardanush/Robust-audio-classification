import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch import Tensor

__all__ = ["log_amp_mel_spectrogram", "FixedLength"]


def log_amp_mel_spectrogram():
    """
    Using original sampling_rate, n_fft, hop_length, n_mels from the CRNN paper
    """
    return nn.Sequential(
        FixedLength(length=176400),
        T.MelSpectrogram(sample_rate=12000, n_fft=512, hop_length=256, n_mels=96),
        T.AmplitudeToDB()
    )


class FixedLength(torch.nn.Module):
    """
    Trim or pad waveform to a fixed length.
    Args:
        length (int): The desired length of the waveform
    """
    __constants__ = ['length']

    def __init__(self, length: int = 176400) -> None:
        super(FixedLength, self).__init__()
        self.length = length

    def forward(self, waveform: Tensor) -> Tensor:
        out = torch.zeros([1, self.length])
        if waveform.numel() < self.length:
            out[:, :waveform.numel()] = waveform
        else:
            out = waveform[:, :self.length]
        return out
