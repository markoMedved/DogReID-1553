# Import available model architectures
# from .video_resnet_reid import VideoResNetReID
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID


def build_model(cfg):
    """
    Factory function that builds the correct model
    based on the configuration file.
    """

    if cfg.model == "dinov2":
        # Using the "reg" variant which is more robust to background artifacts
        model = DINOv2ReID(variant="vitb14_reg")

    elif cfg.model == "vit":
        # Standard Vision Transformer video model
        model = VideoViT()

    elif cfg.model == "swin":
        # Swin Transformer video backbone
        model = VideoSwin()

    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    return model