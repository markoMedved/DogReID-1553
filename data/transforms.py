import torchvision.transforms as T
import torch


class ViTVideoTransform:
    """
    Frame-wise transform for ViT-based video models.
    Expects clip of shape (T, C, H, W) and returns (T, 3, 224, 224).
    """

    def __init__(self):

        self.frame_tf = T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])# data/transforms.py
from torchvision import transforms

class ViTVideoTransform:
    def __init__(self):
        self.frame_tf = transforms.Compose([
            transforms.ToPILImage(),         # 1. Convert NumPy array to PIL
            transforms.Resize((224, 224)),   # 2. Resize PIL image
            transforms.ToTensor(),           # 3. Convert PIL to Tensor (0-1)
            transforms.Normalize(            # 4. Standardize
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])

    def __call__(self, clip):
        # clip is a list or array of NumPy frames
        return torch.stack([self.frame_tf(frame) for frame in clip])

    def __call__(self, clip):

        # Apply transform frame-by-frame
        return torch.stack([self.frame_tf(frame) for frame in clip])