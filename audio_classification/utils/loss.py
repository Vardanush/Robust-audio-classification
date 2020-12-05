import torch
import torch.nn.functional as F

__all__ = ["loss", "label_smoothing_cross_entropy"]

def get_loss(loss_fn='cross_entropy'):
    if loss_fn=='cross_entropy':
        return F.cross_entropy
    elif loss_fn == 'label_smoothing_cross_entropy':
        return label_smoothing_cross_entropy
    else:
        raise Exception("Loss function not found. Please specify a valid loss function.")
    

def label_smoothing_cross_entropy(preds, targets, weight=None, epsilon=0.01):
    """
    Label smoothing from the paper "Rethinking the Inception Architecture for Computer Vision"
    """
    log_preds = F.log_softmax(preds, dim=-1)
    # calcualte (weighted) average of the negative log_probs for all classes, then average over the batch
    if weight is not None:
        smooth_loss = (-1.0 * torch.mv(log_preds, weight)/weight.sum(dim=0)).mean()
    else:
        smooth_loss = (-1.0 * log_preds.sum(dim=-1)/(preds.size()[-1])).mean() 
    nll_loss = F.nll_loss(log_preds, targets, weight=weight)
    loss = (1 - epsilon) * nll_loss + epsilon * smooth_loss    
    return loss