import warnings
import yaml
import logging
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from audio_classification.data import UrbanSoundDataset
from audio_classification.model import LitCRNN
from argparse import ArgumentParser


def get_dataloader(cfg, transform=None):
    logger = logging.getLogger(__name__)

    if cfg.DATASET.NAME == "UrbanSounds8K":
        folds = list(range(1, 11))
        val_folds = [cfg.DATASET.VAL_FOLD]
        train_folds = [fold for fold in folds if fold not in val_folds]

        # create train and test sets using chosen transform
        train_set = UrbanSoundDataset(cfg, train_folds, transform=transform)
        val_set = UrbanSoundDataset(cfg, val_folds, transform=transform)
    # TODO: create train and test for BMW dataset
    else:
        raise ValueError("Unknown dataset: {}".format(cfg.DATASET.NAME))

    logger.info("Train set size: {}".format(str(len(train_set))))
    logger.info("Train set size: {}".format(str(len(train_set))))

    train_loader = DataLoader(train_set, batch_size=cfg.DATALOADER.BATCH_SIZE,
                              shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKDERS,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.DATALOADER.BATCH_SIZE,
                            num_workers=cfg.DATALOADER.NUM_WORKDERS,
                            pin_memory=True)

    return train_loader, val_loader


def get_transform(cfg):
    return None


def get_model(cfg):
    if cfg.MODEL.NAME == "LitCRNN":
        model = LitCRNN()
    # TODO: create model for other classifiers
    else:
        raise ValueError("Unknown model: {}".format(cfg.MODEL.NAME))
    return model


def do_train(cfg):
    logger = logging.getLogger(__name__)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info("Training on device {}".format(device))

    train_loader, val_loader = get_dataloader(cfg, transform=get_transform(cfg))
    tb_logger = pl_loggers.TensorBoardLogger(cfg.SOLVER.LOG_PATH)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=cfg.CHECKPOINT.SAVE_PATH,
        filename=cfg.CHECKPOINT.SAVE_NAME + '-{epoch:02d}-{val_acc:.3f}',
        save_top_k=cfg.CHECKPOINT.SAVE_TOP_K,
        mode='max'
    )

    model = get_model(cfg)
    trainer = pl.Trainer(gpus=cfg.SOLVER.NUM_GPUS,
                         min_epochs=cfg.SOLVER.MIN_EPOCH,
                         max_epochs=cfg.SOLVER.MAX_EPOCH,
                         progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback],
                         logger=tb_logger)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    parser = ArgumentParser()
    parser.add_argument('--layer_1_dim', type=int, default=128)
    args = parser.parse_args()
    # TODO: take config file name as args
    with open("config.yml", "r") as yml_file:
        cfg = yaml.load(yml_file)
    do_train(cfg)
