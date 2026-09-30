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

    model_type = getattr(cfg, "backbone", getattr(cfg, "model", None))
    model_type_str = str(model_type).lower() if model_type else ""

    oa_names = {
        "bot", "oa_bot", "openanimals_bot",
        "agw", "oa_agw", "openanimals_agw",
        "sbs", "oa_sbs", "openanimals_sbs",
        "mgn", "oa_mgn", "openanimals_mgn",
        "arbase", "oa_arbase", "openanimals_arbase",
        "arbase_mb", "oa_arbase_mb", "arbase_mgn",
    }

    # OpenAnimals architectures (SBS, AGW, MGN, ARBase, OA_BoT)
    if model_type_str in oa_names:
        from .openanimals_models import OpenAnimalsVideoModel
        return OpenAnimalsVideoModel(
            model_name=model_type_str,
            pooling_type=getattr(cfg, "pooling_type", "attention"),
            num_classes=getattr(cfg, "num_classes", 0),
            chunk_size=getattr(cfg, "chunk_size", 32)
        )

    # --- Re-ID Method Routing (BoT / TransReID) ---
    if getattr(cfg, "reid_method", None) in ("bot", "transreid"):
        return VideoReID(cfg)

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
