from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch

__all__ = ["UrbanSoundDataset"]


class UrbanSoundDataset(Dataset):
    """
    Wrapper for the UrbanSound8K dataset.

    Parameters:
        csv_path (str): path to the UrbanSound8K csv file
        file_path (str): path to the UrbanSound8k audio files
        folder_list (list): list of folders to use in the dataset
    """

    def __init__(self, csv_path, file_path, folder_list, length=176400, transform=None):
        csv_data = pd.read_csv(csv_path)
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

        self.file_path = file_path
        self.folder_list = folder_list
        self.length = length
        self.transform = transform

    def __getitem__(self, index):
        # format the file path and load the file
        path = self.file_path + "fold" + str(self.folders[index]) + "/" + self.file_names[index]
        # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
        sound = torchaudio.load(path, out=None, normalization=True)
        # UrbanSound8K uses two channels, this will convert them to one
        sound_data = torch.mean(sound[0], dim=0, keepdim=True)
        # pad or trim data to be the same length
        temp_data = torch.zeros([1, self.length])
        if sound_data.numel() < self.length:
            temp_data[:, :sound_data.numel()] = sound_data
        else:
            temp_data = sound_data[:, :self.length]
        sound_data = temp_data

        if self.transform:
            sound_formatted = self.transform(sound_data)
        else:
            sound_formatted = sound_data

        return sound_formatted, self.labels[index]

    def __len__(self):
        return len(self.file_names)
