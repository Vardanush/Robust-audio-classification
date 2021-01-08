import torch
from audio_classification.tools import get_dataloader, get_transform
from audio_classification.model import lit_m11, lit_m18, LitCRNN
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml
import warnings

def get_model(cfg, checkpoint_path, class_weights, map_location):
    if cfg["MODEL"]["NAME"] == "LitCRNN":
        model = LitCRNN.load_from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            cfg=cfg,
                            map_location=map_location, 
                            class_weights=torch.tensor(class_weights).to(device='cuda')
                        )
    elif cfg["MODEL"]["NAME"] == "LitM18":
        model = lit_m18.load_from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            cfg=cfg,
                            map_location=map_location, 
                            class_weights=torch.tensor(class_weights).to(device='cuda')
                        )
    elif cfg["MODEL"]["NAME"] == "LitM11":
        model = lit_m11.load_from_checkpoint(
                            checkpoint_path=checkpoint_path,
                            cfg=cfg,
                            map_location=map_location, 
                            class_weights=torch.tensor(class_weights).to(device='cuda')
                        )
    else:
        raise ValueError("Unknown model: {}".format(cfg["MODEL"]["NAME"]))
    return model

def do_test(configs, checkpoint_path):

    _, _, test_loader, class_weights = get_dataloader(configs, trial_hparams = None, transform=get_transform(configs))
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    model = get_model(configs, checkpoint_path, class_weights, map_location)
    
    trainer = pl.Trainer(gpus=configs["SOLVER"]["NUM_GPUS"])
    trainer.test(model, test_dataloaders=test_loader)
    
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument('--config', default="/nfs/students/winter-term-2020/project-1/project-1/audio_classification/configs/m11_bmw.yaml")
    parser.add_argument('--path', default="/nfs/students/winter-term-2020/project-1/project-1/weights/m11-fold7-epoch=130-val_acc=0.934.ckpt")
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configs = yaml.load(config_file)
    do_test(configs, args.path)

