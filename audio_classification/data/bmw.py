from torch.utils.data import Dataset
import torchaudio
import torch
import os

__all__ = ["BMWDataset"]


class BMWDataset(Dataset):
    """
    Wrapper for the BMW dataset.
    """

    def __init__(self, cfg, transform=None):
        super().__init__()
        self.classes, self.class_to_idx = self._find_classes(cfg["DATASET"]["FOLDER_PATH"])
        self.audios, self.labels = self.make_dataset(
            directory=cfg["DATASET"]["FOLDER_PATH"],
            class_to_idx=self.class_to_idx
        )
        self.transform = transform

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

    def __len__(self):
        length = len(self.audios)
        return length
    
    def __getitem__(self, index):
        data_dict = None
        audio = torchaudio.load(self.audios[index], out=None, normalization=True)
        label = self.labels[index]
        if self.transform:
            audio = self.transform(audio)
        data_dict = {"audio": audio , "label": label}
        
        return data_dict
    
# Add transform classes here if required!
