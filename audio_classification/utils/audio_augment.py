import os
import torch
import torchaudio
import augment
import numpy as np

__all__ = ["get_augment"]

    
def get_augment(cfg=None):     
    # randomly transform the audio clip with pitch shift, reverberation and addtive noise
    random_delay_seconds = lambda: np.random.uniform(0, 1)
    random_tempo_ratio = lambda: np.random.uniform(0.75, 1.25)
    random_pitch_shift = lambda: np.random.randint(-200, +200)
    random_room_size = lambda: np.random.randint(0, 101)
    # noise_generator = lambda: torch.zeros_like(x).uniform_()
    combination = augment.EffectChain() \
        .delay(random_delay_seconds) \
        .tempo("-q", random_tempo_ratio) \
        .pitch("-q", random_pitch_shift).rate(sr) \
        .reverb(50, 50, random_room_size).channels(1) \
    #     .additive_noise(noise_generator, snr=50)

    runner = ChainRunner(combination)
    return runner


class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'rate': 48000}  # rate of the sample (for BMW:48000)

        target_info = {'channels': 1,
                       'rate': 48000}

        y = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y
    