"""Re-ID heads.

BNNeck   Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person
         Re-Identification", CVPRW 2019 / IEEE TMM 2020. arXiv:1903.07071
JPM      He et al., "TransReID: Transformer-based Object Re-Identification",
         ICCV 2021, Sec. 3.3. arXiv:2102.04378

Both operate on token sequences of shape (N, 1 + L, D) with the class token at
index 0, so they apply to any ViT-family backbone.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    if m.__class__.__name__.find("Linear") != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class BNNeckHead(nn.Module):
    """Batch-normalization neck.

    The triplet loss is applied to the feature before the bottleneck and the
    identity loss to the feature after it, which is the configuration reported
    as strongest in the BoT ablation. Retrieval uses the post-BN feature.
    """

    def __init__(self, dim: int, num_classes: int = 0):
        super().__init__()

        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(dim, num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, feat):
        """Returns (triplet feature, retrieval feature, logits or None)."""
        feat_bn = self.bottleneck(feat)

        cls_score = None
        if self.training and self.num_classes > 0:
            cls_score = self.classifier(feat_bn)

        return feat, F.normalize(feat_bn, dim=-1), cls_score


def shuffle_patches(x, shift: int = 5, groups: int = 2):
    """Shift and patch-shuffle a token sequence of shape (N, L, D).

    Tokens are rolled by `shift` positions and then interleaved across `groups`
    strided sub-sequences, so that each jigsaw group covers the whole image
    rather than one contiguous spatial band.
    """
    N, L, D = x.shape

    x = torch.roll(x, shifts=-shift, dims=1)

    keep = (L // groups) * groups
    head, tail = x[:, :keep], x[:, keep:]
    head = head.view(N, groups, keep // groups, D).permute(0, 2, 1, 3).reshape(N, keep, D)

    return torch.cat([head, tail], dim=1)


class JPM(nn.Module):
    """Jigsaw Patch Module.

    The penultimate token sequence feeds two branches. The global branch uses
    the backbone's own final block; the local branch uses an independent copy of
    it, applied to k shuffled patch groups that are each re-prefixed with the
    class token. Returns one global plus k local features.
    """

    def __init__(self, last_block, norm, k: int = 4, shift: int = 5, shuffle_groups: int = 2):
        super().__init__()

        self.global_block = last_block
        self.global_norm = norm

        # The local branch does not share weights with the global branch
        self.local_block = copy.deepcopy(last_block)
        self.local_norm = copy.deepcopy(norm)

        self.k = k
        self.shift = shift
        self.shuffle_groups = shuffle_groups

    def forward(self, x):
        # --- Global Branch ---
        x_global = self.global_norm(self.global_block(x))
        feats = [x_global[:, 0]]

        # --- Local Branches ---
        cls_token, patches = x[:, :1], x[:, 1:]
        patches = shuffle_patches(patches, self.shift, self.shuffle_groups)

        L = patches.shape[1]
        per_group = L // self.k

        for i in range(self.k):
            start = i * per_group
            end = L if i == self.k - 1 else (i + 1) * per_group
            group = torch.cat([cls_token, patches[:, start:end]], dim=1)
            group = self.local_norm(self.local_block(group))
            feats.append(group[:, 0])

        return feats
