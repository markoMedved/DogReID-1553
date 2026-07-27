# --- Import Available Model Architectures ---
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID
from .miewid_builder import MiewIDReID
from .megadescriptor_builder import MegaDescriptor
from .reid_model import VideoReID


def build_model(cfg):
    """
    Factory function to instantiate the requested model architecture
    based on the provided configuration parameters.
    """

    # MiewID is a fixed pretrained baseline; it bypasses the BoT/TransReID
    # heads because the checkpoint already provides a trained embedding.
    model_type = getattr(cfg, "backbone", getattr(cfg, "model", None))
    if model_type == "miewid":
        return MiewIDReID(chunk_size=getattr(cfg, "chunk_size", 16))

    # --- Re-ID Method Routing (BoT / TransReID) ---
    if getattr(cfg, "reid_method", None) in ("bot", "transreid"):
        return VideoReID(cfg)

    if model_type == "dinov2":
        # Initializes DINOv2 with registers (vitb14_reg)
        return DINOv2ReID(variant="vitb14_reg")

    elif model_type == "vit":
        # Initializes a standard Vision Transformer adapted for video processing
        return VideoViT()

    elif model_type == "swin":
        # Initializes a Swin Transformer backbone for hierarchical video feature extraction
        return VideoSwin()

    elif model_type == "megadescriptor":
        # Wildlife-re-ID-pretrained Swin used as a standalone extractor (no BoT).
        return MegaDescriptor(
            variant=getattr(cfg, "megadescriptor_variant", "hf-hub:BVRA/MegaDescriptor-L-224"),
            chunk_size=getattr(cfg, "chunk_size", 8),
        )

    else:
        # Fallback for unsupported or misspelled model configurations
        raise ValueError(f"Unknown model architecture requested: {model_type}")