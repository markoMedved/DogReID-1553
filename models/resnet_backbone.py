import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):

    def __init__(self, name="resnet50", pretrained=True):

        super().__init__()

        if name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            self.out_dim = 2048


        layers = list(model.children())[:-1]

        self.backbone = nn.Sequential(*layers)


    def forward(self, x):

        x = self.backbone(x)

        x = x.flatten(1)

        return x
    
    