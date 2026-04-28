# --- Import Available Model Architectures ---
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID


def build_model(cfg):
    """
    Factory function to instantiate the requested model architecture
    based on the provided configuration parameters.
    """

    # --- Model Selection Routing ---

    if cfg.model == "dinov2":
        # Initializes DINOv2 with registers (vitb14_reg)
        model = DINOv2ReID(variant="vitb14_reg")

    elif cfg.model == "vit":
        # Initializes a standard Vision Transformer adapted for video processing
        model = VideoViT()

    elif cfg.model == "swin":
        # Initializes a Swin Transformer backbone for hierarchical video feature extraction
        model = VideoSwin()

    else:
        # Fallback for unsupported or misspelled model configurations
        raise ValueError(f"Unknown model architecture requested: {cfg.model}")

    return model