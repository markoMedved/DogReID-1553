import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, labels):

        dist = torch.cdist(embeddings, embeddings)

        N = dist.size(0)

        mask_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        mask_neg = ~mask_pos

        hardest_pos = (dist * mask_pos.float()).max(dim=1)[0]

        dist_neg = dist.clone()
        dist_neg[mask_pos] = 1e9
        hardest_neg = dist_neg.min(dim=1)[0]

        target = torch.ones_like(hardest_neg)

        loss = self.ranking_loss(hardest_neg, hardest_pos, target)

        return loss