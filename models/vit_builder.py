import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

def build_vit():
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads = nn.Identity()
    return model

class VideoViT(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False):
        super().__init__()
        self.vit = build_vit()
        
        # BNNeck: helps balance Triplet and ID loss
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck.bias.requires_grad_(False)  # no bias for BNNeck
        
        # Classifier head for ID Loss (Label Smoothing)
        self.classifier = nn.Linear(768, num_classes, bias=False)

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        # 1. Feature Extraction
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.vit(x)          
            feats = feats.view(B, T, -1).mean(dim=1) # (B, 768)
        else:
            feats = self.vit(x)

        # 2. Bottleneck
        # During training, Triplet Loss uses 'feats' (before BN)
        # While ID Loss and Evaluation use 'bn_feats' (after BN)
        bn_feats = self.bottleneck(feats)

        if self.training:
            logits = self.classifier(bn_feats)
            return feats, logits  # Return both for dual loss
        
        return bn_feats  # During eval, return normalized/BN features