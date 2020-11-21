import numpy as np

__all__ = ["calc_weights"]


def calc_weights(sets, cfg = None):
    """
    Calculate class weights as described in the paper https://arxiv.org/abs/1707.03237
    
    params: sets is the type of dataset 
            cfg from yaml files
    """
    labels = []
    lengths = []
    for i in range(0, len(sets)):
        sample = sets[i]
        lengths.append(sample[0].shape[1])
        labels.append(sample[1])
    
    labels = np.array(labels, dtype=np.float32)
    num_labelled = np.array([
            np.sum(np.equal(labels, c))
            for c in range(cfg["MODEL"]["NUM_CLASSES"])
        ], dtype=np.float32)
    
    class_weight = (labels.size / (num_labelled + 0.00001)).astype(np.float32) # multiplied by labels.size to prevent numerical issues
    
    if cfg["DATASET"]["WEIGHT"] == "SQUARED":
        class_weight = class_weight**2
    
    return class_weight
    
    

