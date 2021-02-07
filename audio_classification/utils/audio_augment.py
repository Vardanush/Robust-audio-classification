import os
import torch
import torchaudio
import augment
import numpy as np
import random

__all__ = ["get_available_noises", "get_random_noise", "noise_augment", "random_augment",
           "uniform_augment", "gaussian_augment"]


def get_available_noises(path='/nfs/students/winter-term-2020/project-1/MUSAN/free-sound/'):
    """
    Get all noises available in MUSAN background audio clips.
    """
    annot = os.path.join(path, 'ANNOTATIONS')
    if os.path.isfile(annot):
        with open(annot) as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        background_noises = [path + name + '.wav' for name in content[1:]]
        return background_noises
    else:
        raise ValueError("Not a valid annotation file for noises: {}".format(annot))


def get_random_noise(background_noises, out_shape, out_sr=48000):
    """
    Randomly select a noise from available noise clips.
    """
    noise_idx=random.randint(0, len(background_noises)-1)
    noise_frames = int(out_shape/3)+1
    random_offset = random.randint(0, 10000)
    noise, sr_noise = torchaudio.backend.sox_io_backend.load(background_noises[noise_idx], frame_offset=random_offset, num_frames=noise_frames)
    transform = torchaudio.transforms.Resample(sr_noise, out_sr)
    noise = transform(noise)
    noise = noise[:, :out_shape]
    return noise


def apply_augment(x, chain):
    """
    Apply a chain of audio augmentation to a clip.
    :param x: audio clip
    :param chain: augment function to be applied to the audio
    :return:
        x: augmented audio
    """
    src_info = {'channels': x.size(0),  # number of channels
            'rate': 48000}  # rate of the sample (for BMW:48000)
    target_info = {'channels': 1,
                   'rate': 48000}
    try:
        y = chain.apply(
            x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y
    except AssertionError:
        # Noise and signal shapes are incompatible if the noise clip is shorter than the signal
        # In this case, add no augmention
#         print("Random selected noise is incompatible with shape of input: {}".format(x.shape[1]))
        return x
    

def noise_augment(x, extra_noises=None):
    """
    Additive real noise from the MUSAN audio datasets
    """
    noise_generator = lambda: get_random_noise(extra_noises, x.shape[1])
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=15)
    return apply_augment(x, combination)


def uniform_augment(x, extra_noises=None):
    """
    Additive uniform noise to audio clip with randomly sampled snr
    """
    noise_generator = lambda: torch.zeros_like(x).uniform_()
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=30)
    return apply_augment(x, combination)


def gaussian_augment(x, extra_noises=None):
    """
    Additive Gaussian noise to audio clip with randomly sampled snr.
    """
    noise_generator = lambda: torch.randn(x.shape)
    combination = augment.EffectChain() \
        .additive_noise(noise_generator, snr=30) 
    return apply_augment(x, combination)


def random_augment(x, extra_noises=None):
    """
    Randomly transform the audio clip with time stretch and pitch shift.
    """
    random_tempo_ratio = lambda: np.random.uniform(0.75, 1.25)
    random_pitch_shift = lambda: np.random.randint(-200, +200)
    combination = augment.EffectChain() \
        .tempo("-q", random_tempo_ratio) \
        .pitch("-q", random_pitch_shift).rate(48000)
    return apply_augment(x, combination)


# map the inputs to the function blocks
augment_options = {"random" : random_augment,
           "uniform" : uniform_augment,
           "gaussian" : gaussian_augment,
           "noise": noise_augment,
}
