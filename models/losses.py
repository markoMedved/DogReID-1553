"""Native Re-ID loss functions.

Pure PyTorch implementations of:
- Batch-hard Triplet Loss with hard example mining
- Weighted Regularized Triplet Loss (AGW) with softmax example mining
- Soft-margin Triplet Loss (margin=0.0)
- Label-smoothed Cross Entropy Loss
- Circle Loss / CircleSoftmax modulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance between row vectors of x and y."""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Cosine distance between row vectors of x and y."""
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2.0 - 2.0 * torch.matmul(x, y.t())
    return dist


def softmax_weights(dist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax weights for weighted example mining (AGW)."""
    mask_b = mask.bool()
    dist_masked = dist.masked_fill(~mask_b, -float('inf'))
    max_v = torch.max(dist_masked, dim=1, keepdim=True)[0]
    diff = (dist - max_v).masked_fill(~mask_b, -float('inf'))
    exp_diff = torch.exp(diff)
    Z = torch.sum(exp_diff, dim=1, keepdim=True) + 1e-6
    return exp_diff / Z


def hard_example_mining(dist_mat: torch.Tensor, is_pos: torch.Tensor, is_neg: torch.Tensor):
    """Find hardest positive and hardest negative sample for each anchor."""
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)
    return dist_ap, dist_an


def weighted_example_mining(dist_mat: torch.Tensor, is_pos: torch.Tensor, is_neg: torch.Tensor):
    """Weighted positive and negative sample mining (AGW)."""
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)
    return dist_ap, dist_an


def triplet_loss(
    embedding: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.3,
    norm_feat: bool = False,
    hard_mining: bool = True
) -> torch.Tensor:
    """Batch-hard or weighted triplet loss with margin ranking or soft margin.

    Args:
        embedding: (N, D) feature embeddings
        targets: (N,) class labels
        margin: triplet margin. If 0.0, soft-margin loss log(1 + exp(d_ap - d_an)) is used.
        norm_feat: if True, uses cosine distance; else euclidean distance.
        hard_mining: if True, uses batch-hard mining; else weighted mining (AGW).
    """
    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    N = dist_mat.size(0)
    is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0.0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if torch.isinf(loss) or torch.isnan(loss):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)

    return loss


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Cross-entropy loss with label smoothing."""
    if eps <= 0.0:
        return F.cross_entropy(logits, targets)

    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        smoothed_targets = torch.full_like(log_probs, eps / (num_classes - 1))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1.0 - eps)

    loss = (-smoothed_targets * log_probs).sum(dim=1)
    return loss.mean()


class CircleSoftmax(nn.Module):
    """CircleSoftmax loss layer from Sun et al., 'Circle Loss: A Unified Perspective

    of Pair Similarity Optimization', CVPR 2020.
    Modulates positive and negative margins based on similarity values.
    """

    def __init__(self, num_classes: int, scale: float = 64.0, margin: float = 0.35):
        super().__init__()
        self.num_classes = num_classes
        self.s = scale
        self.m = margin

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Modulate cosine logits with Circle Loss margins.

        Args:
            logits: (N, C) cosine similarities F.linear(normalize(x), normalize(w))
            targets: (N,) class labels
        """
        alpha_p = torch.clamp_min(-logits.detach() + 1.0 + self.m, min=0.0)
        alpha_n = torch.clamp_min(logits.detach() + self.m, min=0.0)
        delta_p = 1.0 - self.m
        delta_n = self.m

        index = torch.where(targets != -1)[0]
        m_hot = torch.zeros(index.size(0), logits.size(1), device=logits.device, dtype=logits.dtype)
        m_hot.scatter_(1, targets[index, None], 1.0)

        logits_p = alpha_p * (logits - delta_p)
        logits_n = alpha_n * (logits - delta_n)

        out_logits = logits.clone()
        out_logits[index] = logits_p[index] * m_hot + logits_n[index] * (1.0 - m_hot)

        neg_index = torch.where(targets == -1)[0]
        if len(neg_index) > 0:
            out_logits[neg_index] = logits_n[neg_index]

        out_logits.mul_(self.s)
        return out_logits

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, scale={self.s}, margin={self.m}"
