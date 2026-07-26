# --- Import Available Model Architectures ---
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID
from .reid_model import VideoReID


def build_model(cfg):
    """
    Factory function to instantiate the requested model architecture
    based on the provided configuration parameters.
    """

    # --- Re-ID Method Routing (BoT / TransReID) ---
    if getattr(cfg, "reid_method", None) in ("bot", "transreid"):
        return VideoReID(cfg)
    
    # --- Legacy / Standalone Model Architectures ---
    model_type = getattr(cfg, "backbone", getattr(cfg, "model", None))

    if model_type == "dinov2":
        # Initializes DINOv2 with registers (vitb14_reg)
        return DINOv2ReID(variant="vitb14_reg")

    elif model_type == "vit":
        # Initializes a standard Vision Transformer adapted for video processing
        return VideoViT()

    elif model_type == "swin":
        # Initializes a Swin Transformer backbone for hierarchical video feature extraction
        return VideoSwin()

    else:
        # Fallback for unsupported or misspelled model configurations
        raise ValueError(f"Unknown model architecture requested: {model_type}")