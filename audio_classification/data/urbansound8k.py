"""
UrbanSound8K dataset.
"""
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch
from audio_classification.utils import get_available_noises
from audio_classification.utils.audio_augment import augment_options
__all__ = ["UrbanSoundDataset"]


class UrbanSoundDataset(Dataset):
    """
    Wrapper for the UrbanSound8K dataset.
    """

    def __init__(self, cfg, folder_list, transform=None, augment='none'):
        csv_data = pd.read_csv(cfg["DATASET"]["ANNOTATION_PATH"])
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csv_data)):
            if csv_data.iloc[i, 5] in folder_list:
                self.file_names.append(csv_data.iloc[i, 0])
                self.labels.append(csv_data.iloc[i, 6])
                self.folders.append(csv_data.iloc[i, 5])

        self.file_path = cfg["DATASET"]["FILE_PATH"]
        self.folder_list = folder_list
        self.transform = transform
        
        #for agumentation
        self.augment = augment
        print("Data augmentation for US8K dataset: {}".format(self.augment))
        if self.augment =='noise':
            self.background_noises = get_available_noises(path=cfg['DATASET']['NOISE_PATH'])
        else:
            self.background_noises = None

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        sound = torchaudio.load(path, out=None, normalization=True)
        # UrbanSound8K uses two channels, this will convert them to one
        sound_data = torch.mean(sound[0], dim=0, keepdim=True)

        # for agumentation
        if self.augment!='none':
            audio = augment_options[self.augment](sound_data, extra_noises = self.background_noises)
            
        if self.transform:
            sound_formatted = self.transform(sound_data)
        else:
            sound_formatted = sound_data

        return sound_formatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)
