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
        ])

    def __call__(self, clip):

        # Apply transform frame-by-frame
        return torch.stack([self.frame_tf(frame) for frame in clip])