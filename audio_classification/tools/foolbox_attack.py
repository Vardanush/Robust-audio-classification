from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from foolbox.devutils import flatten
from foolbox.devutils import atleast_kd
from foolbox.types import Bounds
from foolbox.models.base import Model
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import l1, l2, linf
from foolbox.attacks.base import FixedEpsilonAttack
from foolbox.attacks.base import T
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import raise_if_kwargs

from pprint import pprint
import os
import time
import types
import yaml
import torch
from audio_classification.tools import do_train, get_dataloader, get_model, get_transform
from audio_classification.tools.train_net import collate
from audio_classification.model import lit_m11, LitCRNN, SmoothClassifier
from audio_classification.data import BMWDataset, UrbanSoundDataset
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD, FGM, FGSM
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import IPython.display as ipd
from scipy.io.wavfile import write


def _get_loss_fn(self, model: Model, labels: ep.Tensor, original_lengths=None) -> Callable[[ep.Tensor], ep.Tensor]:
    # can be overridden by users
    def loss_fn(inputs: ep.Tensor, original_lengths) -> ep.Tensor:
        logits = model(inputs, original_lengths)
        return ep.crossentropy(logits, labels).sum()

    return loss_fn


def _value_and_grad(
    # can be overridden by users
    self,
    loss_fn: Callable[[ep.Tensor], ep.Tensor],
    x: ep.Tensor,
    original_lengths: ep.Tensor,
) -> Tuple[ep.Tensor, ep.Tensor]:
    return ep.value_and_grad(loss_fn, x, original_lengths)


def _run(
    self,
    model: Model,
    inputs: T,
    criterion: Union[Misclassification, TargetedMisclassification, T],
    *,
    epsilon: float,
    **kwargs: Any,
) -> T:
#     raise_if_kwargs(kwargs)
    x0, restore_type = ep.astensor_(inputs)
    criterion_ = get_criterion(criterion)
    original_lengths = kwargs['original_lengths']
    del inputs, criterion, kwargs

    # perform a gradient ascent (targeted attack) or descent (untargeted attack)
    if isinstance(criterion_, Misclassification):
        gradient_step_sign = 1.0
        classes = criterion_.labels
    elif hasattr(criterion_, "target_classes"):
        gradient_step_sign = -1.0
        classes = criterion_.target_classes  # type: ignore
    else:
        raise ValueError("unsupported criterion")

    loss_fn = self.get_loss_fn(model, classes)

    if self.abs_stepsize is None:
        stepsize = self.rel_stepsize * epsilon
    else:
        stepsize = self.abs_stepsize

    if self.random_start:
        x = self.get_random_start(x0, epsilon)
        x = ep.clip(x, *model.bounds)
    else:
        x = x0

    for _ in range(self.steps):
        _, gradients = self.value_and_grad(loss_fn, x, original_lengths) #!!!!
        gradients = self.normalize(gradients, x=x, bounds=model.bounds)
        x = x + gradient_step_sign * stepsize * gradients
        x = self.project(x, x0, epsilon)
        x = ep.clip(x, *model.bounds)

    return restore_type(x)


def attack_model(project_dir, config_path, pretrained_path, title, project="BMW", attack_type = 'linf', max_radius=10, save_folder='attack_results/'):
#     device = torch.device('cpu')
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    torch.backends.cudnn.enabled = False
    with open(os.path.join(project_dir, config_path), "r") as config_file:
        configs = yaml.load(config_file)
    configs["ATTACK"]=True
    print(configs)

    # use test/validattion set
    if project=="BMW":
        val_set = BMWDataset(configs, [11], transform=get_transform(configs)) # actually the test set
        num_batches = 3
    elif project=="UrbanSound8k":
        val_set = UrbanSoundDataset(configs, [10], transform=get_transform(configs))
        num_batches = 40
    val_loader = DataLoader(val_set, batch_size=20, shuffle=False,
                                    num_workers=configs["DATALOADER"]["NUM_WORKERS"],
                                    pin_memory=True, collate_fn = collate)
    del val_set

    # Get the upper bound and lower bound on the values of the data, to be used as constraint in PGD
    lower_bounds = []
    upper_bounds = []
    for step, (x, y, seq_lens) in enumerate(val_loader):    
        upper_bounds.append(torch.max(x))
        lower_bounds.append(torch.min(x))
    lower_bound = min(lower_bounds)
    upper_bound = max(upper_bounds)
    print("Range of the input data is (%f, %f)" %(lower_bound, upper_bound))

    path_to_checkpoint = os.path.join(project_dir, pretrained_path)
    
    # Get the class weights, used in reloading the model
    if configs['DATASET']['WEIGHT']=='NORMAL':
        weight = torch.tensor([28.9047, 14.8049,  4.5985,  2.4675,  4.4632, 19.5806]).to(device=device)
    elif configs['DATASET']['WEIGHT']=='SQUARED':
        weight = torch.tensor([835.4845, 219.1843,  21.1461,   6.0885,  19.9205, 383.4014]).to(device=device)
    else:
        weight = None

    if configs["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        base_classifier = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)
        model = SmoothClassifier.load_from_checkpoint(checkpoint_path=path_to_checkpoint, cfg=configs, map_location=device, class_weights=weight, base_classifier = base_classifier.to(device=device))
    else:    
        model = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)

    fmodel = PyTorchModel(model, bounds=(lower_bound, upper_bound), device=device)
    
    # set up Fast Gradient Attack
    torch.cuda.empty_cache()
    if attack_type == 'linf':
        attack = FGSM()
        epsilons = np.linspace(0.0, max_radius, num=20)
    elif attack_type == 'l2':
        attack = FGM()
        epsilons = np.linspace(0.0, max_radius, num=50)

    attack.run = types.MethodType(_run, attack)
    attack.get_loss_fn = types.MethodType(_get_loss_fn, attack)
    attack.value_and_grad = types.MethodType(_value_and_grad, attack)

    # Evaluate robust robustness
    start_time = time.perf_counter()    
    robust_accuracy = []
    for epsilon in epsilons:
        it = iter(val_loader)
        is_adv_all = []
        for n in range(0, num_batches):
            batch = next(it)
            clips = batch[0].to(device)
            labels = batch[1].to(device)
            lengths = batch[2].to(device)   # used only for CRNN
            torch.cuda.empty_cache()
            raw, clipped, is_adv = attack(fmodel, clips, labels, epsilons=epsilon, original_lengths=lengths)
            preds = []
            for i, clip in enumerate(clipped):
                x = torch.unsqueeze(clip, 0)
                length = torch.unsqueeze(lengths[i], 0)
                out = model(x.to(device), length)
                preds.append(torch.argmax(out, dim=1).item())
            is_adv = preds!=labels.cpu().numpy()
            is_adv_all.append(is_adv)
        is_adv_all = np.concatenate(is_adv_all)
        robust_accuracy.append(1-sum(1*is_adv_all)/len(is_adv_all))
        del it
    end_time = time.perf_counter()
    print(f"Generated attacks in {end_time - start_time:0.2f} seconds")
    print(robust_accuracy)

    plt.title( attack_type + " Fast Gradient Attack")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.1)
    plt.plot(epsilons, robust_accuracy)
    plt.savefig(save_folder + title + '-'+ attack_type +'-' + str(max_radius) + '.png')

    
def attack_model_for_randomize_smoothing(project_dir, config_path, pretrained_path, title, project="BMW", attack_type = 'linf', max_radius=10, save_folder='attack_results/'):
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    torch.backends.cudnn.enabled = False
    with open(os.path.join(project_dir, config_path), "r") as config_file:
        configs = yaml.load(config_file)
    configs["ATTACK"]=True
    print(configs)

    # use test/validattion set
    if project=="BMW":
        val_set = BMWDataset(configs, [11], transform=get_transform(configs)) # actually the test set
        num_batches = 3
    elif project=="UrbanSound8k":
        val_set = UrbanSoundDataset(configs, [10], transform=get_transform(configs))
        num_batches = 40
    val_loader = DataLoader(val_set, batch_size=20, shuffle=False,
                                    num_workers=configs["DATALOADER"]["NUM_WORKERS"],
                                    pin_memory=True, collate_fn = collate)
    del val_set

    # Get the upper bound and lower bound on the values of the data, to be used as constraint in PGD
    lower_bounds = []
    upper_bounds = []
    for step, (x, y, seq_lens) in enumerate(val_loader):    
        upper_bounds.append(torch.max(x))
        lower_bounds.append(torch.min(x))
    lower_bound = min(lower_bounds)
    upper_bound = max(upper_bounds)
    print("Range of the input data is (%f, %f)" %(lower_bound, upper_bound))

    path_to_checkpoint = os.path.join(project_dir, pretrained_path)
    
    # Get the class weights, used in reloading the model
    if configs['DATASET']['WEIGHT']=='NORMAL':
        weight = torch.tensor([28.9047, 14.8049,  4.5985,  2.4675,  4.4632, 19.5806]).to(device=device)
    elif configs['DATASET']['WEIGHT']=='SQUARED':
        weight = torch.tensor([835.4845, 219.1843,  21.1461,   6.0885,  19.9205, 383.4014]).to(device=device)
    else:
        weight = None

    if configs["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        base_classifier = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)
        model = SmoothClassifier.load_from_checkpoint(checkpoint_path=path_to_checkpoint, cfg=configs, map_location=device, class_weights=weight, base_classifier = base_classifier.to(device=device))
    else:    
        model = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)

    fmodel = PyTorchModel(model, bounds=(lower_bound, upper_bound), device=device)
    
    # set up Fast Gradient Attack
    torch.cuda.empty_cache()
    attack = FGSM()

    attack.run = types.MethodType(_run, attack)
    attack.get_loss_fn = types.MethodType(_get_loss_fn, attack)
    attack.value_and_grad = types.MethodType(_value_and_grad, attack)
    epsilons = np.linspace(0.0, max_radius, num=20)

    # Evaluate robust robustness
    start_time = time.perf_counter()    
    robust_accuracy = []
    for epsilon in epsilons:
        it = iter(val_loader)
        is_adv_all = []
        for n in range(0, num_batches):
            batch = next(it)
            clips = batch[0].to(device)
            labels = batch[1].to(device)
            lengths = batch[2].to(device)   # used only for CRNN
            torch.cuda.empty_cache()
            raw, clipped, is_adv = attack(fmodel, clips, labels, epsilons=epsilon, original_lengths=lengths)
            preds = []
            for i, clip in enumerate(clipped):
                x = torch.unsqueeze(clip, 0)
                preds.append(model.predict(x.to(device), seq_len=lengths[i], num_samples=50, alpha=0.05, batch_size=1))
            is_adv = preds!=labels.cpu().numpy()
            is_adv_all.append(is_adv)
        is_adv_all = np.concatenate(is_adv_all)
        robust_accuracy.append(1-sum(1*is_adv_all)/len(is_adv_all))
        del it
    end_time = time.perf_counter()
    print(f"Generated attacks in {end_time - start_time:0.2f} seconds")
    print(robust_accuracy)

    plt.title("L-inf Fast Gradient Attack")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.1)
    plt.plot(epsilons, robust_accuracy)
    plt.savefig(save_folder + title + '-linf-' + str(max_radius) + '.png')

    
def attack_model_per_class(project_dir, config_path, pretrained_path, project="BMW", attack_type = 'linf', epsilon=10):
#     device = torch.device('cpu')
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    torch.backends.cudnn.enabled = False
    with open(os.path.join(project_dir, config_path), "r") as config_file:
        configs = yaml.load(config_file)
    configs["ATTACK"]=True
    print(configs)

    # use test/validattion set
    if project=="BMW":
        val_set = BMWDataset(configs, [11], transform=get_transform(configs)) # actually the test set
        num_batches = 6
    elif project=="UrbanSound8k":
        val_set = UrbanSoundDataset(configs, [10], transform=get_transform(configs))
        num_batches = 40
    val_loader = DataLoader(val_set, batch_size=10, shuffle=False,
                                    num_workers=configs["DATALOADER"]["NUM_WORKERS"],
                                    pin_memory=True, collate_fn = collate)
    del val_set

    # Get the upper bound and lower bound on the values of the data, to be used as constraint in PGD
    lower_bounds = []
    upper_bounds = []
    for step, (x, y, seq_lens) in enumerate(val_loader):    
        upper_bounds.append(torch.max(x))
        lower_bounds.append(torch.min(x))
    lower_bound = min(lower_bounds)
    upper_bound = max(upper_bounds)
    print("Range of the input data is (%f, %f)" %(lower_bound, upper_bound))

    path_to_checkpoint = os.path.join(project_dir, pretrained_path)
    
    # Get the class weights, used in reloading the model
    if configs['DATASET']['WEIGHT']=='NORMAL':
        weight = torch.tensor([28.9047, 14.8049,  4.5985,  2.4675,  4.4632, 19.5806]).to(device=device)
    elif configs['DATASET']['WEIGHT']=='SQUARED':
        weight = torch.tensor([835.4845, 219.1843,  21.1461,   6.0885,  19.9205, 383.4014]).to(device=device)
    else:
        weight = None

    if configs["MODEL"]["CRNN"]["RANDOMISED_SMOOTHING"] == True:
        base_classifier = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)
        model = SmoothClassifier.load_from_checkpoint(checkpoint_path=path_to_checkpoint, cfg=configs, map_location=device, class_weights=weight, base_classifier = base_classifier.to(device=device))
    else:    
        model = LitCRNN.load_from_checkpoint(path_to_checkpoint, cfg=configs, class_weights=weight, strict=False, map_location=device)

    fmodel = PyTorchModel(model, bounds=(lower_bound, upper_bound), device=device)
    
    # set up Fast Gradient Attack
    torch.cuda.empty_cache()
    attack = FGSM()

    attack.run = types.MethodType(_run, attack)
    attack.get_loss_fn = types.MethodType(_get_loss_fn, attack)
    attack.value_and_grad = types.MethodType(_value_and_grad, attack)

    # Evaluate robust robustness per class   
    it = iter(val_loader)
    is_adv_all = []
    preds_all = []
    labels_all = []
    for n in range(0, num_batches):
        batch = next(it)
        clips = batch[0].to(device)
        labels = batch[1].to(device)
        lengths = batch[2].to(device)   # used only for CRNN
        torch.cuda.empty_cache()
        raw, clipped, is_adv = attack(fmodel, clips, labels, epsilons=epsilon, original_lengths=lengths)
        preds = []
        for i, clip in enumerate(clipped):
            x = torch.unsqueeze(clip, 0)
            length = torch.unsqueeze(lengths[i], 0)
            out = model(x.to(device), length)
            preds.append(torch.argmax(out, dim=1).item())
        is_adv = preds!=labels.cpu().numpy()
        is_adv_all.append([str(x) for x in is_adv])
        preds_all.append(preds)
        labels_all.append(labels.cpu().numpy())
    is_adv_all = np.concatenate(is_adv_all)
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    del it
    class_count = {}
    is_adv_count = {}
    for i, label in enumerate(labels_all):
        if label not in class_count:
            class_count[label] = {}
        if preds_all[i] not in class_count[label]:
            class_count[label][preds_all[i]] = 1
        else:
            class_count[label][preds_all[i]] += 1
        
        if label not in is_adv_count:
            is_adv_count[label] = {}
        if is_adv_all[i] not in is_adv_count[label]:
            is_adv_count[label][is_adv_all[i]] = 1
        else:
            is_adv_count[label][is_adv_all[i]] += 1
    print("Attack raidus: %f" % epsilon)
    print("Per class frequncy of prediction class from adversarial samples:")
    pprint(dict(class_count))
    print("Per class frequncy of sucessful attacks from adversarial samples:")
    pprint(dict(is_adv_count))
    return class_count, is_adv_count
    