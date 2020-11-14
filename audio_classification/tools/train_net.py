import warnings
import yaml
import logging
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from audio_classification.data import UrbanSoundDataset
from audio_classification.data import BMWDataset
from audio_classification.model import LitCRNN
from audio_classification.model import LitDeepCNN, lit_m18, lit_m11
from audio_classification.utils import audio_transform
from argparse import ArgumentParser


def adjust_len(seq, length):
    '''
    to enable sequences of variable lengths in a batch we pad them
    '''
    out = torch.zeros([1, length])
    if seq.shape[-1] < length:
            out[:, :seq.numel()] = seq
    else:
            out = seq[:, :length]
    return out

def my_collate(batch):
    '''
    From https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    Keep in mind for RNN take hidden state corresponding to the last non padded input value
    Use torch.nn.utils.rnn.pack_padded_sequence
    
    '''
    data = [item['audio'][0] for item in batch] # not including sr in the batch
    max_length = max([item.shape[1] for item in data])
  
    data =[adjust_len(item, max_length) for item in data]
    data=torch.stack(data)
    target = [item['label'] for item in batch]
    target = torch.LongTensor(target)

    return [data, target]


def get_dataloader(cfg, transform=None):
    logger = logging.getLogger(__name__)

    if cfg["DATASET"]["NAME"] == "UrbanSounds8K":
        folds = list(range(1, 11))
        val_folds = [cfg["DATASET"]["VAL_FOLD"]]
        train_folds = [fold for fold in folds if fold not in val_folds]

        # create train and test sets using chosen transform
        train_set = UrbanSoundDataset(cfg, train_folds, transform=transform)
        val_set = UrbanSoundDataset(cfg, val_folds, transform=transform)
    elif cfg["DATASET"]["NAME"] == "BMW":
        sets = BMWDataset(cfg, transform=transform) # train, val, test split in the ratio 8:1:1
        train_samples = len(sets)*80 // 100
        val_samples = len(sets)*10 // 100
        test_samples = len(sets) - train_samples - val_samples
        
        train_set, val_set, test_set = torch.utils.data.random_split(sets, [train_samples, val_samples, test_samples])
        
    else:
        raise ValueError("Unknown dataset: {}".format(cfg["DATASET"]["NAME"]))

    logger.info("Train set size: {}".format(str(len(train_set))))
    logger.info("Val set size: {}".format(str(len(val_set))))

    if cfg["MODEL"]["NAME"] == "LitM18" or cfg["MODEL"]["NAME"] == "LitM11":
        collate_fn = my_collate
    else:
        collate_fn = None
        
        
    if cfg["MODEL"]["SANITY_CHECK"] == 1: # TODO: sanity check is defined in m18_bmw yaml. Add it to other yaml files
        train_loader = train_loader # TODO: change it for the data loaders to have only 1 sample and other elif statements for 2,5,10 samples
        
        
    train_loader = DataLoader(train_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
                                  shuffle=True, num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                  pin_memory=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
                                num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                pin_memory=True, collate_fn = collate_fn)
        
    if cfg["DATASET"]["NAME"] == "BMW":
        test_loader = DataLoader(test_set, batch_size=cfg["DATALOADER"]["BATCH_SIZE"],
                            num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                            pin_memory=True, collate_fn = collate_fn)
    else:
        test_loader = None
        
    return train_loader, val_loader, test_loader


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


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--config', type=int, default="m18_bmw.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configs = yaml.load(config_file)
    do_train(configs)
