import os
import torch
import torchaudio
import augment
import numpy as np
import random

__all__ = ["random_augment"]


background_noises = get_avilabke_noises()

def get_available_noises(noise_path = '../datasets/MUSAN/free-sound/'):
    with open(os.path.join(noise_path, 'ANNOTATIONS'))as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    background_noises = [name + '.wav' for name in content[1:]]
    return backgound_noises

def get_random_noise(background_noises, out_shape, out_sr=48000):
    noise_idx=random.randint(0, len(background_noises)-1)
    noise_frames = int(out_shape/3)+1
    random_offset = random.randint(0, 10000)
    noise, sr_noise = torchaudio.backend.sox_io_backend.load(os.path.join(noise_path, background_noises[noise_idx]), frame_offset=random_offset, num_frames=noise_frames)
    transform = torchaudio.transforms.Resample(sr_noise, out_sr)
    noise = transform(noise)
    noise = noise[:, :out_shape]
    return noise

def noise_augment(x, cfg=None):
    # additive real noise from the MUSAN audio datasets
    noise_generator = lambda: get_random_noise(background_noises, x.shape[1])
combination = augment.EffectChain() \
    .additive_noise(noise_generator, snr=15)
    runner = ChainRunner(combination)
    return runner

def uniform_augment(x, cfg=None):
    # additive uniform noise to audio clip with randomly sampled snr
    noise_generator = lambda: torch.zeros_like(x).uniform_()
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=5)   # roughly 0.75 * signal + 0.25 * noise 
    runner = ChainRunner(combination)
    return runner 

def gaussian_augment(x, cfg=None):
    # additive Gaussian noise to audio clip with randomly sampled snr
    noise_generator = lambda: torch.randn(x)
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=10) # roughly 0.9 * signal + 0.1 * noise

    runner = ChainRunner(combination)
    return runner  

def random_augment(cfg=None):     
    # randomly transform the audio clip with pitch shift, reverberation and addtive noise
    random_tempo_ratio = lambda: np.random.uniform(0.75, 1.25)
    random_pitch_shift = lambda: np.random.randint(-200, +200)
#     random_room_size = lambda: np.random.randint(0, 101)
    # random_delay_seconds = lambda: np.random.uniform(0, 1)
    # noise_generator = lambda: torch.zeros_like(x).uniform_()
    combination = augment.EffectChain() \
        .tempo("-q", random_tempo_ratio) \
        .pitch("-q", random_pitch_shift).rate(48000) \
#         .reverb(50, 50, random_room_size).channels(1) \
    #     .delay(random_delay_seconds) \
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
    