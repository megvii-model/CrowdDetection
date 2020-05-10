import megengine as mge
import megengine.functional as F
from megengine.core import Tensor

def softmax_loss(pred, label, ignore_label=-1):
    max_pred = F.zero_grad(pred.max(axis=1, keepdims=True))
    pred -= max_pred
    log_prob = pred - F.log(F.exp(pred).sum(axis=1, keepdims=True))
    mask = 1 - F.equal(label, ignore_label)
    vlabel = label * mask
    loss = -(F.indexing_one_hot(log_prob, vlabel, 1) * mask)
    return loss

def smooth_l1_loss(pred, target, beta: float):
    abs_x = F.abs(pred - target)
    in_mask = abs_x < beta
    out_mask = 1 - in_mask
    in_loss = 0.5 * abs_x ** 2 / beta
    out_loss = abs_x - 0.5 * beta
    loss = in_loss * in_mask + out_loss * out_mask
    return loss.sum(axis=1)

