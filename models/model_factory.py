# from .video_resnet_reid import VideoResNetReID
from .vit_builder import VideoViT

from .vit_builder import VideoViT

def build_model(cfg):
    if cfg.model == "vit":
        # num_classes should be 744 based on your logs
        model = VideoViT(num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    return model