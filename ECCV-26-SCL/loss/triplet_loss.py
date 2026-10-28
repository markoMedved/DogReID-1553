import torch
from torch import nn


def normalize(x, axis=-1):
    return x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels):
    assert dist_mat.dim() == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    n = dist_mat.size(0)
    is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
    dist_ap = torch.max(dist_mat[is_pos].contiguous().view(n, -1), 1, keepdim=True)[0]
    dist_an = torch.min(dist_mat[is_neg].contiguous().view(n, -1), 1, keepdim=True)[0]
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class TripletLoss:
    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False, **kwargs):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap = dist_ap * (1.0 + self.hard_factor)
        dist_an = dist_an * (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an
