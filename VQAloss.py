import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsort

class VQALoss(nn.Module):
    def __init__(self, scale, loss_type='mixed', m=None):
        super(VQALoss, self).__init__()
        self.loss_type = loss_type
        self.scale = scale
        self.m = m

    def forward(self, y_pred, y):
        relative_score, mapped_score, aligned_score = y_pred
        if self.loss_type == 'plcc':
            loss = [loss_accuracy(mapped_score[d], y[d]) for d in range(len(y))]
        elif self.loss_type == 'srcc':
            loss = [loss_monotonicity(relative_score[d], y[d]) for d in range(len(y))]
        elif self.loss_type == 'plcc+srcc':
            loss = [loss_accuracy(mapped_score[d], y[d]) + loss_monotonicity(relative_score[d], y[d]) for d in range(len(y))]
        else: # default l1
            loss = [F.l1_loss(aligned_score[d], y[d]) / self.scale[d] for d in range(len(y))]

        sum_loss = sum([torch.exp(lossi) * lossi for lossi in loss]) / sum([torch.exp(lossi) for lossi in loss])
        return sum_loss

def loss_accuracy(y_pred, y):
    """prediction accuracy related loss"""
    assert y_pred.size(0) > 1
    return (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2

def loss_monotonicity(y_pred, y, **kw):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1
    pred = torch.t(y_pred)
    pred = torchsort.soft_rank(pred, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = torch.t(y)
    target = torchsort.soft_rank(target, **kw)
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()
