import torch.nn.functional as F

__all__ = ["label_smoothing_cross_entropy"]

def label_smoothing_cross_entropy(preds, targets, epsilon: float = 0.1, reduction: str = 'mean'):
    """
    Label smoothing from the paper "Rethinking the Inception Architecture for Computer Vision"
    """
    # TODO: use class weights in smoothing
    Z = preds.size()[-1] # number of classes
    log_preds = F.log_softmax(preds, dim=-1)
    smooth_loss = -1.0*log_preds.sum(dim=-1).mean()
    nll_loss = F.nll_loss(log_preds, targets)
    loss = nll_loss * (1 - epsilon) + epsilon * (smooth_loss / Z)
    return loss