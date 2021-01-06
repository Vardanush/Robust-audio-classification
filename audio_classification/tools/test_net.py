import torch
from audio_classification.tools import get_dataloader, get_transform
from audio_classification.model import lit_m11, lit_m18, LitCRNN, SmoothClassifier
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml
import warnings
from typing import Dict
from tqdm.autonotebook import tqdm

def get_model(cfg, checkpoint_path, class_weights, map_location):
    
    if class_weights is not None:
        weights  = torch.tensor(class_weights).to(device='cuda')
    else:
        weights = None
    
    if cfg["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        
        if cfg["MODEL"]["NAME"] == "LitCRNN":
            base_classifier = LitCRNN(cfg=cfg, class_weights=weights)
        elif cfg["MODEL"]["NAME"] == "LitM18":
            base_classifier = lit_m18(cfg, weights)
        elif cfg["MODEL"]["NAME"] == "LitM11":
            base_classifier = lit_m11(cfg, weights)
        else:
            raise ValueError("Unknown model: {}".format(cfg["MODEL"]["NAME"]))
        

        model = SmoothClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path, cfg=cfg, map_location=map_location, class_weights=weights, base_classifier = base_classifier.to(device='cuda'))
                            
    else: 
       
        if cfg["MODEL"]["NAME"] == "LitCRNN":
            model = LitCRNN.load_from_checkpoint(
                                checkpoint_path=checkpoint_path,
                                cfg=cfg,
                                map_location=map_location, 
                                class_weights=weights
                            )
        elif cfg["MODEL"]["NAME"] == "LitM18":
            model = lit_m18.load_from_checkpoint(
                                checkpoint_path=checkpoint_path,
                                cfg=cfg,
                                map_location=map_location, 
                                class_weights=weights
                            )
        elif cfg["MODEL"]["NAME"] == "LitM11":
            model = lit_m11.load_from_checkpoint(
                                checkpoint_path=checkpoint_path,
                                cfg=cfg,
                                map_location=map_location, 
                                class_weights=weights
                            )
        else:
            raise ValueError("Unknown model: {}".format(cfg["MODEL"]["NAME"]))
        
    return model

def evaluate_robustness_smoothing(model, test_loader,
                                  num_samples_1: int = 1000, num_samples_2: int = 10000,
                                  alpha: float = 0.05, certification_batch_size: float = 5000) -> Dict:
    """
    Evaluate the robustness of a smooth classifier based on the input base classifier via randomized smoothing.
    Parameters
    ----------
    base_classifier: Classifier
        The input base classifier to use in the randomized smoothing process.
    test_dataloader: Dataloader
        The data used for evaluation
    num_samples_1: int
        The number of samples used to determine the most likely class.
    num_samples_2: int
        The number of samples used to perform the certification.
    alpha: float
        The desired confidence level that the top class is indeed the most likely class. E.g. alpha=0.05 means that
        the expected error rate must not be larger than 5%.
    certification_batch_size: int
        The batch size to use during the certification, i.e. how many noise samples to classify in parallel.

    Returns
    -------
    Dict containing the following keys:
        * abstains: int. The number of times the smooth classifier abstained, i.e. could not certify the input sample to
                    the desired confidence level.
        * false_predictions: int. The number of times the prediction could be certified but was not correct.
        * correct_certified: int. The number of times the prediction could be certified and was correct.
        * avg_radius: float. The average radius for which the predictions could be certified.

    """
    abstains = 0
    false_predictions = 0
    correct_certified = 0
    radii = []
    
    for batch in tqdm(test_loader):
        x, y, seq_len = batch # Here batch size is 1 
        x = x.cuda()
        pred_class, radius = model.certify(x, num_samples_1, num_samples_2, alpha=alpha,
                                           batch_size=certification_batch_size, seq_len=seq_len)
        if pred_class == y:
            correct_certified += 1
            radii.append(radius)
        elif pred_class == SmoothClassifier.ABSTAIN:
            abstains += 1
            radii.append(0.)
        elif pred_class != y:
            false_predictions += 1
            radii.append(0.)

    avg_radius = torch.tensor(radii).mean().item()
    return dict(abstains=abstains, false_predictions=false_predictions, correct_certified=correct_certified,
                avg_radius=avg_radius)


def do_test(configs, checkpoint_path):
    # Makesure the batch size of dataloader is 1
    configs["DATALOADER"]["BATCH_SIZE"] = 1

    _, _, test_loader, class_weights = get_dataloader(configs, trial_hparams = None, transform=get_transform(configs))
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'

    model = get_model(configs, checkpoint_path, class_weights, map_location)
    
    if configs["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        result = evaluate_robustness_smoothing(model, test_loader,num_samples_1=int(60), num_samples_2=int(60), alpha=0.05, certification_batch_size=int(10))
        print(result)
        
    else:
        
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
