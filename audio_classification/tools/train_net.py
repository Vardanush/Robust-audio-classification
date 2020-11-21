import warnings
import yaml
import logging
import torch
from torch.utils.data import DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from audio_classification.data import UrbanSoundDataset
from audio_classification.data import BMWDataset
from audio_classification.model import LitCRNN
from audio_classification.model import LitDeepCNN, lit_m18, lit_m11
from audio_classification.utils import audio_transform
from argparse import ArgumentParser

def my_collate(batch):
    '''
    From https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    Keep in mind for RNN take hidden state corresponding to the last non padded input value
    Use torch.nn.utils.rnn.pack_padded_sequence
    
    '''
    data = [item[0] for item in batch] # not including sr in the batch
    target = [item[1] for item in batch]
    length = [item.shape[-1] for item in data]
    
    max_length = max(length)
    data =[torch.nn.functional.pad(item, (0, max_length - item.shape[-1])) for item in data]
    data=torch.stack(data)
    
    target = torch.LongTensor(target)
    length = torch.LongTensor(length)

    return [data, target, length]

def get_transform(cfg):
    if cfg["MODEL"]["NAME"] == "LitCRNN":
        transform = audio_transform.log_amp_mel_spectrogram(cfg=cfg)
    elif cfg["MODEL"]["NAME"] == "LitM18" or cfg["MODEL"]["NAME"] == "LitM11":
        if cfg["DATASET"]["NAME"] == "UrbanSounds8K":
            transform = torchaudio.transforms.Resample(44100, 8000)
        else:
            transform = None
    else:
        transform = None
    return transform

def get_dataloader(cfg, transform=None):
    folds = list(range(1, 11))
    val_folds = [cfg["DATASET"]["VAL_FOLD"]]
    train_folds = [fold for fold in folds if fold not in val_folds]
    
    if cfg["DATASET"]["NAME"] == "UrbanSounds8K":
        # create train and test sets using chosen transform
        train_set = UrbanSoundDataset(cfg, train_folds, transform=transform)
        val_set = UrbanSoundDataset(cfg, val_folds, transform=transform)
    elif cfg["DATASET"]["NAME"] == "BMW":
        train_set = BMWDataset(cfg, train_folds, transform=transform)
        val_set = BMWDataset(cfg, val_folds, transform=transform)   
    else:
        raise ValueError("Unknown dataset: {}".format(cfg["DATASET"]["NAME"]))

    if cfg["MODEL"]["NAME"] == "LitM18" or cfg["MODEL"]["NAME"] == "LitM11":
        collate_fn = my_collate
    else:
        collate_fn = None
        
        
    if cfg["MODEL"]["SANITY_CHECK"] == 1: # TODO: sanity check is defined in m18_bmw yaml. Add it to other yaml files
        train_loader = train_loader # TODO: change it for the data loaders to have only 1 sample and other elif statements for 2,5,10 samples
        
        
    collate_fn = my_collate    
    train_loader = DataLoader(train_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
                                  shuffle=True, num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                  pin_memory=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
                                num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                pin_memory=True, collate_fn = collate_fn)
# TODO: add back test_loader        
#     if cfg["DATASET"]["NAME"] == "BMW":
#         test_loader = DataLoader(test_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
#                             num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
#                             pin_memory=True, collate_fn = collate_fn)
#     else:
#         test_loader = None
    test_loader = None
    return train_loader, val_loader, test_loader


def get_model(cfg):
    if cfg["MODEL"]["NAME"] == "LitCRNN":
        model = LitCRNN(cfg)
    elif cfg["MODEL"]["NAME"] == "LitM18":
        model = lit_m18(cfg)
    elif cfg["MODEL"]["NAME"] == "LitM11":
        model = lit_m11(cfg)
    else:
        raise ValueError("Unknown model: {}".format(cfg["MODEL"]["NAME"]))
    return model


def do_train(cfg):
    logger = logging.getLogger(__name__)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info("Training on device {}".format(device))

    train_loader, val_loader, test_loader = get_dataloader(cfg, transform=get_transform(cfg))
    tb_logger = pl_loggers.TensorBoardLogger(cfg["SOLVER"]["LOG_PATH"])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=cfg["CHECKPOINT"]["SAVE_PATH"],
        filename=cfg["CHECKPOINT"]["SAVE_NAME"] + '-{epoch:02d}-{val_acc:.3f}',
        save_top_k=cfg["CHECKPOINT"]["SAVE_TOP_K"],
        mode='max'
    )

    model = get_model(cfg)
    trainer = pl.Trainer(gpus=cfg["SOLVER"]["NUM_GPUS"],
                         min_epochs=cfg["SOLVER"]["MIN_EPOCH"],
                         max_epochs=cfg["SOLVER"]["MAX_EPOCH"],
                         progress_bar_refresh_rate=10,
                         callbacks=[checkpoint_callback],
                         logger=tb_logger)

    trainer.fit(model, train_loader, val_loader)
    
    return test_loader


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--config', type=int, default="m18_bmw.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configs = yaml.load(config_file)
    do_train(configs)
