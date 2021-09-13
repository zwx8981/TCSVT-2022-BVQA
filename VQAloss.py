import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsort

class VQALoss(nn.Module):
    def __init__(self, scale, loss_type='mixed', m=None):
        super(VQALoss, self).__init__()
        self.loss_type = loss_type
        self.scale = scale
        self.m = m #

    def forward(self, y_pred, y):
        relative_score, mapped_score, aligned_score = y_pred
        if self.loss_type == 'mixed':
            loss = [loss_a(mapped_score[d], y[d]) + loss_m(relative_score[d], y[d]) +
                        F.l1_loss(aligned_score[d], y[d]) / self.scale[d] for d in range(len(y))]
        elif self.loss_type == 'correlation' or self.loss_type == 'rank+plcc':
            loss = [loss_a(mapped_score[d], y[d]) + loss_m(relative_score[d], y[d]) for d in range(len(y))]
        elif self.loss_type == 'rank':
            loss = [loss_m(relative_score[d], y[d]) for d in range(len(y))]
        elif self.loss_type == 'plcc':
            loss = [loss_a(mapped_score[d], y[d]) for d in range(len(y))]
        elif self.loss_type == 'rank+l1':
            loss = [loss_m(relative_score[d], y[d]) + F.l1_loss(aligned_score[d], y[d]) / self.scale[d] for d in range(len(y)) for d in range(len(y))]
        elif self.loss_type == 'plcc+l1':
            loss = [loss_a(relative_score[d], y[d]) + F.l1_loss(aligned_score[d], y[d]) / self.scale[d] for d in range(len(y)) for d in range(len(y))]
        elif 'naive' in self.loss_type:
            aligned_scores = torch.cat([(aligned_score[d]-self.m[d])/self.scale[d] for d in range(len(y))])
            ys = torch.cat([(y[d]-self.m[d])/self.scale[d] for d in range(len(y))])
            if self.loss_type == 'naive0':
                return F.l1_loss(aligned_scores, ys) #
            # return loss_a(aligned_scores, ys) + loss_m(aligned_scores, ys) + F.l1_loss(aligned_scores, ys)
            return loss_a(aligned_scores, ys) + spearmanr(torch.t(aligned_scores), torch.t(ys))
        elif self.loss_type == 'srcc':
            loss = [spearmanr(torch.t(relative_score[d]), torch.t(y[d])) for d in range(len(y))]
        elif self.loss_type == 'plcc+srcc':
            loss = [loss_a(mapped_score[d], y[d]) + spearmanr(torch.t(relative_score[d]), torch.t(y[d])) for d in range(len(y))]
        else: # default l1
            loss = [F.l1_loss(aligned_score[d], y[d]) / self.scale[d] for d in range(len(y))]

        sum_loss = sum([torch.exp(lossi) * lossi for lossi in loss]) / sum([torch.exp(lossi) for lossi in loss])
        return sum_loss


def loss_m(y_pred, y):
    """prediction monotonicity related loss"""
    assert y_pred.size(0) > 1  #
    return torch.sum(F.relu((y_pred-y_pred.t()) * torch.sign((y.t()-y)))) / y_pred.size(0) / (y_pred.size(0)-1)

def loss_a(y_pred, y):
    """prediction accuracy related loss"""
    assert y_pred.size(0) > 1  #
    return (1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]) / 2

def loss_a2(y_pred, y):
    """prediction accuracy related loss"""
    assert y_pred.size(0) > 1  #
    return 1 - torch.cosine_similarity(y_pred.t() - torch.mean(y_pred), y.t() - torch.mean(y))[0]

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return 1 - (pred * target).sum()
