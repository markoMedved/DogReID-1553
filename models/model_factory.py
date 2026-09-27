# --- Import Available Model Architectures ---
from .vit_builder import VideoViT
from .swin_builder import VideoSwin
from .dinov2_builder import DINOv2ReID
from .convnetxt_builder import VideoConvNeXt
from .miewid_builder import MiewIDReID
from .megadescriptor_builder import MegaDescriptor
from .reid_model import VideoReID


def build_model(cfg):
    """
    Factory function to instantiate the requested model architecture
    based on the provided configuration parameters.
    """

    # --- Re-ID Method Routing (BoT / TransReID) ---
    if getattr(cfg, "reid_method", None) in ("bot", "transreid"):
        return VideoReID(cfg)

    model_type = getattr(cfg, "backbone", getattr(cfg, "model", None))
    if model_type:
        model_type_str = str(model_type).lower()
        if model_type_str.startswith("oa_") or model_type_str.startswith("openanimals_"):
            from .openanimals_models import OpenAnimalsVideoModel
            return OpenAnimalsVideoModel(
                model_name=model_type_str,
                pooling_type=getattr(cfg, "pooling_type", "attention"),
                num_classes=getattr(cfg, "num_classes", 0),
                chunk_size=getattr(cfg, "chunk_size", 32)
            )

    # MiewID is a fixed pretrained baseline; it bypasses the BoT/TransReID
    if model_type == "miewid":
        return MiewIDReID(chunk_size=getattr(cfg, "chunk_size", 16))

    if model_type == "dinov2":
        # Initializes DINOv2 with registers (vitb14_reg)
        return DINOv2ReID(
            variant=getattr(cfg, "dinov2_variant", "vitb14_reg"), 
            num_classes=getattr(cfg, "num_classes", 0), 
            chunk_size=getattr(cfg, "chunk_size", 32),
            pooling_type=getattr(cfg, "pooling_type", "attention")
        )

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

    elif model_type in ("convnetxt", "convnext"):
        return VideoConvNeXt()

    else:
        # Fallback for unsupported or misspelled model configurations
        raise ValueError(f"Unknown model architecture requested: {model_type}")
