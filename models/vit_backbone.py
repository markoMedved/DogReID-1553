import torch
import torch.nn as nn
import timm


class ViTBackbone(nn.Module):

    def __init__(self, name="vit_base_patch16_224", pretrained=True):

        super().__init__()

        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0  # removes classifier
        )

        self.out_dim = self.model.num_features

    def forward(self, x):

        x = self.model(x)

        return x