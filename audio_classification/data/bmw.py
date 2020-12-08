from torch.utils.data import Dataset
import torchaudio
import torch
import os
import numpy as np
import pandas as pd
import os.path
from sklearn.model_selection import StratifiedKFold 

__all__ = ["BMWDataset"]


class BMWDataset(Dataset):
    """
    Wrapper for the BMW dataset.
    """

    def __init__(self, cfg, folder_list, transform=None, augment=None):
        super().__init__()
        self.annotation_path = cfg["DATASET"]["ANNOTATION_PATH"]
        if os.path.isfile(self.annotation_path):
            csv_data = pd.read_csv(self.annotation_path)
        else:
            print("No existing annotation meta for BMW dataset. Generating...")
            classes, class_to_idx = self._find_classes(cfg["DATASET"]["FOLDER_PATH"])
            audios, labels = self.make_dataset(
                directory=cfg["DATASET"]["FOLDER_PATH"],
                class_to_idx=class_to_idx
            )
            csv_data = self.stratifed_kfold(np.array(audios), np.array(labels), 
                                            n_split=11, shuffle=True, random_state=1,
                                           save_path=self.annotation_path)
            csv_data = csv_data.astype({'fold': 'int', 'classID': 'int'})
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csv_data)):
            if csv_data.iloc[i, 1] in folder_list:
                self.file_names.append(csv_data.iloc[i, 0])
                self.labels.append(csv_data.iloc[i, 2])
                self.folders.append(csv_data.iloc[i, 1])

        self.folder_list = folder_list
        self.transform = transform
        self.augment = augment # a Callable that applies data augmentation
        

    @staticmethod
    def _find_classes(directory):
        """
        Finds the class folders in a dataset
        :param directory: root directory of the dataset
        :returns: (classes, class_to_idx), where
          - classes is the list of all classes found
          - class_to_idx is a dict that maps class to label
        """
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(directory, class_to_idx):
        """
        Create the bmw audio dataset by preparaing a list of samples
        :param directory: root directory of the dataset
        :param class_to_idx: A dict that maps classes to labels
        :returns: (audios, labels) where:
            - audios is a list containing paths to all wav files in the dataset
            - labels is a list containing one label per audio
        """
        audios, labels = [], []
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        for cls in sorted (classes):
            current_dir = os.path.join(directory, cls)
            samples_per_class = len([name for name in os.listdir (current_dir) if os.path.isfile (os.path.join (current_dir, name))])
            for i in os.scandir(current_dir):
                if i.name.endswith('.wav'):
                    audio_path = os.path.join(current_dir, i)
                audios.append(audio_path)
                labels.append(class_to_idx[cls])

        assert len(audios) == len(labels)
        return audios, labels
    
    @staticmethod
    def stratifed_kfold(X, y, n_split=10, shuffle=True, random_state=1, save_path=None):
        kfold = StratifiedKFold(n_splits=n_split, shuffle=shuffle, random_state=random_state)
        meta = []
        fold = 1
        for train_ix, test_ix in kfold.split(X, y):
            train_X, test_X = X[train_ix], X[test_ix]
            train_y, test_y = y[train_ix], y[test_ix]
            fold_meta = np.vstack((X[test_ix] ,np.full(len(test_y), fold) ,y[test_ix])).T.tolist()
            meta += fold_meta
            fold = fold + 1
        df = pd.DataFrame(meta) # construct data frame and transpose
        df = df.rename(columns={0: "slice_file_name", 1: "fold", 2: "classID"})
        if save_path:
            df.to_csv(save_path, header=True, index=False, sep=',')
        return df

    def __len__(self):
        length = len(self.file_names)
        return length
    
    def __getitem__(self, index):
        sound = torchaudio.load(self.file_names[index], out=None, normalization=True)
        label = self.labels[index]
        audio = sound[0]
        
        if self.augment:
            audio = self.agument(audio)  
        
        if self.transform:
            audio = self.transform(audio)
    
        return audio, label
    
# Add transform classes here if required!
