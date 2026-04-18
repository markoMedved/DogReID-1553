# from .video_resnet_reid import VideoResNetReID
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID

def build_model(cfg):
    if cfg.model == "dinov2":
        # Using the "reg" variant for background artifact suppression
        return DINOv2ReID(variant="vitb14_reg")
    elif cfg.model == "vit":
        return VideoViT()
    elif cfg.model == "swin":
        model = VideoSwin()
    else:
        raise ValueError(f"Unknown model: {cfg.model}")

    return model