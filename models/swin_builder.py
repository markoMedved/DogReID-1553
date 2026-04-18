import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

class TemporalAttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape: (B, T, Dim)
        weights = self.attn(x) # (B, T, 1)
        return (x * weights).sum(dim=1)

class VideoSwin(nn.Module):
    def __init__(self, num_classes=None, chunk_size=8):
        super().__init__()
        # 1. Load Swin V2 Base and rename to 'backbone' for train.py compatibility
        weights = Swin_V2_B_Weights.DEFAULT
        self.backbone = swin_v2_b(weights=weights)
        
        # 2. Remove original classification head
        self.backbone.head = nn.Identity()
        
        # 3. Settings
        self.dim = 1024 # Swin-B output dimension
        self.chunk_size = chunk_size # Helps avoid OOM on MIG slices
        
        # 4. Attention Pooling 
        self.temporal_pool = TemporalAttentionPool(self.dim)
        
        # 5. BN-Neck (Crucial for Triplet Loss)
        self.bn = nn.BatchNorm1d(self.dim)
        self.bn.bias.requires_grad_(False) 

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            
            # --- CHUNKED FORWARD ---
            # Swin-v2-B is memory intensive. Processing in chunks keeps usage stable.
            chunks = torch.split(x, self.chunk_size, dim=0)
            feats = torch.cat([self.backbone(c) for c in chunks], dim=0) 
            
            # Reshape back to (B, T, D) for pooling
            feats = feats.view(B, T, -1) 
            
            # --- ATTENTION POOLING ---
            feats = self.temporal_pool(feats) 
        else:
            # Single image input
            feats = self.backbone(x)

        # Apply BN-Neck and L2 Normalize
        feats = self.bn(feats)
        return F.normalize(feats, p=2, dim=1)