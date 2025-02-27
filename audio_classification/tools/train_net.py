"""
Training Routine.
"""
import warnings
import yaml
import logging
import torch
from torch.utils.data import DataLoader
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from audio_classification.data import UrbanSoundDataset
from audio_classification.data import BMWDataset
from audio_classification.model import LitCRNN, SmoothClassifier, SmoothADV
from audio_classification.model import LitDeepCNN, lit_m18, lit_m11
from audio_classification.utils import audio_transform
from audio_classification.utils import class_weighting
from argparse import ArgumentParser


def collate(batch):
    """
    Pad each batch.
    From https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
    Keep in mind for RNN take hidden state corresponding to the last non padded input value
    Use torch.nn.utils.rnn.pack_padded_sequence.
    """
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
    """
    Get transforms for audio based on model, dataset and configuration.
    :param cfg: model configurations
    :return: torchaudio compatible transform
    """
    if cfg["MODEL"]["NAME"] == "LitCRNN" and not cfg["MODEL"]["CRNN"]["INCLUDE_TRANSFORM"]:
        print("Transformed raw audio into melspectrogram in dataloader.")
        transform = audio_transform.log_amp_mel_spectrogram(cfg=cfg)
    elif cfg["MODEL"]["NAME"] == "LitM18" or cfg["MODEL"]["NAME"] == "LitM11":
        if cfg["DATASET"]["NAME"] == "UrbanSounds8K":
            transform = torchaudio.transforms.Resample(44100, 8000)
        else:
            transform = torchaudio.transforms.Resample(48000, 8000)
    else:
        transform = None
    return transform


def get_dataloader(cfg, trial_hparams=None, transform=None, augment=None):
    """
    Get dataloader based on model configuration.
    :param cfg: model configurations
    :param trial_hparams: parameters for hyperparmeter training
    :param transform: transforms for the audio
    :param augment: whether to add augmentation on the audio.
    :return:
        train_loader: data loader for training
        val_loader: data loader for validation
        test_loader: data loader for test
        class_weights: a list of class weighting to use in training unbalanced data.
    """
    folds = list(range(1, 11))
    val_folds = [cfg["DATASET"]["VAL_FOLD"]]
    test_folds = [11] if cfg["DATASET"]["NAME"] == "BMW" else [10]
    train_folds = [fold for fold in folds if fold not in val_folds and fold not in test_folds]
    print('train_folds', train_folds)
    print('val_folds', val_folds)
    print('test_folds', test_folds)

    if cfg["DATASET"]["NAME"] == "UrbanSounds8K":
        # create train and test sets using chosen transform
        sets = UrbanSoundDataset(cfg, folds, transform=transform)
        train_set = UrbanSoundDataset(cfg, train_folds, transform=transform, augment=augment)
        val_set = UrbanSoundDataset(cfg, val_folds, transform=transform)
        test_set = UrbanSoundDataset(cfg, test_folds, transform=transform)

    elif cfg["DATASET"]["NAME"] == "BMW":
        sets = BMWDataset(cfg, folds, transform=transform)
        train_set = BMWDataset(cfg, train_folds, transform=transform, augment=augment)
        val_set = BMWDataset(cfg, val_folds, transform=transform)
        test_set = BMWDataset(cfg, test_folds, transform=transform)
    else:
        raise ValueError("Unknown dataset: {}".format(cfg["DATASET"]["NAME"]))

    collate_fn = collate
    
    if trial_hparams is not None:
        batch_size = trial_hparams["batch_size"]
    else:
        batch_size = cfg["DATALOADER"]["BATCH_SIZE"]

    train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=True, drop_last = True, num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                  pin_memory=True, collate_fn = collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last = True,
                                num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                pin_memory=True, collate_fn = collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last = True,
                                num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                                pin_memory=True, collate_fn = collate_fn)


    class_weights = class_weighting.calc_weights(sets, cfg)
    return train_loader, val_loader, test_loader, class_weights


def get_model(cfg, weights, device, trial_hparams, train_loader, val_loader):
    """
    :param cfg: model configurations
    :param weights: a list of class weighting to use in training unbalanced data.
    :param trial_hparams: parameters for hyperparmeter training
    :param train_loader: data loader for training
    :param val_loader: data loader for validation
    :return: a model based on model configurations.
    """
    if cfg["MODEL"]["NAME"] == "LitCRNN":
        model = LitCRNN(cfg=cfg, class_weights=weights, trial_hparams=trial_hparams, train_loader=train_loader, val_loader=val_loader)
    elif cfg["MODEL"]["NAME"] == "LitM18":
        model = lit_m18(cfg=cfg, class_weights=weights, trial_hparams=trial_hparams, train_loader=train_loader, val_loader=val_loader)
    elif cfg["MODEL"]["NAME"] == "LitM11":
        model = lit_m11(cfg=cfg, class_weights=weights, trial_hparams=trial_hparams, train_loader=train_loader, val_loader=val_loader)
    else:
        raise ValueError("Unknown model: {}".format(cfg["MODEL"]["NAME"]))

    if cfg["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        model = SmoothClassifier(cfg = cfg, class_weights=weights, base_classifier=model, trial_hparams=trial_hparams, train_loader=train_loader, val_loader=val_loader)
    elif cfg["MODEL"]["CRNN"]["SMOOTH_ADV"] == True:
        model = SmoothADV(cfg = cfg, class_weights=weights, base_classifier=model, device=device, trial_hparams=trial_hparams, train_loader=train_loader, val_loader=val_loader)

    return model


def do_train(cfg):
    """
    Train the model according to model configurations. Save the logs and weights.
    :param cfg: model configurations
    """
    logger = logging.getLogger(__name__)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    logger.info("Training on device {}".format(device))

    transform = get_transform(cfg)
    augment = cfg['DATASET']['AUGMENTATION']
    train_loader, val_loader, test_loader, class_weights = get_dataloader(cfg, transform=transform, augment=augment)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg["SOLVER"]["LOG_PATH"], name=cfg["CHECKPOINT"]["SAVE_NAME"])
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=cfg["CHECKPOINT"]["SAVE_PATH"],
        filename=cfg["CHECKPOINT"]["SAVE_NAME"] + '-{epoch:02d}-{val_acc:.3f}',
        save_top_k=cfg["CHECKPOINT"]["SAVE_TOP_K"],
        mode='max'
    )
    
    if class_weights is not None:
        class_weights = torch.tensor(class_weights).to(device=device)

    trial_hparams = None

    model = get_model(cfg, class_weights, device, trial_hparams, train_loader, val_loader)
    profiler = AdvancedProfiler(output_filename='profile.txt')
    trainer = pl.Trainer(gpus=cfg["SOLVER"]["NUM_GPUS"],
                         min_epochs=cfg["SOLVER"]["MIN_EPOCH"],
                         max_epochs=cfg["SOLVER"]["MAX_EPOCH"],
                         progress_bar_refresh_rate=10,
                         callbacks=[checkpoint_callback],
                         logger=tb_logger,
                         profiler=profiler)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--config', type=int, default="m18_bmw.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configs = yaml.load(config_file)
    do_train(configs)
