

__all__ = ["random_augmentation"]

def random_augmentation(cfg=None):
    """
    """
    x, sr = torchaudio.load(csv_data.loc[idx, 'slice_file_name'])
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
    y = combination.apply(x, src_info={'rate': sr}, target_info={'rate': sr})
    plt.plot(y.cpu().numpy().flatten())
    plt.show()
    ipd.Audio(y, rate=sr)
    