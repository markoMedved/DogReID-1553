"""Re-ID heads.

BNNeck   Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person
         Re-Identification", CVPRW 2019 / IEEE TMM 2020. arXiv:1903.07071
"""

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

    def __init__(self, dim: int, num_classes: int = 0, use_bnneck: bool = True):
        super().__init__()

        self.use_bnneck = use_bnneck
        if use_bnneck:
            self.bottleneck = nn.BatchNorm1d(dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.bottleneck = nn.Identity()

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

