# --- Import Available Model Architectures ---
from .dinov2_builder import DINOv2ReID
from .reid_model import VideoReID


def build_model(cfg):
    """
    Factory function to instantiate the requested model architecture
    based on the provided configuration parameters.
    """

    model_type = getattr(cfg, "backbone", getattr(cfg, "model", None))
    model_type_str = str(model_type).lower() if model_type else ""


    # OpenAnimals architectures (default for bot, agw, sbs, mgn, arbase, and dinov2 variants)
    oa_names = {
        "bot", "oa_bot", "openanimals_bot",
        "dinov2_bot", "oa_dinov2", "oa_dinov2_bot", "openanimals_dinov2_bot",
        "agw", "oa_agw", "openanimals_agw",
        "dinov2_agw", "oa_dinov2_agw", "openanimals_dinov2_agw",
        "sbs", "oa_sbs", "openanimals_sbs",
        "dinov2_sbs", "oa_dinov2_sbs", "openanimals_dinov2_sbs",
        "mgn", "oa_mgn", "openanimals_mgn",
        "arbase", "oa_arbase", "openanimals_arbase",
        "dinov2_arbase", "oa_dinov2_arbase", "openanimals_dinov2_arbase",
        "arbase_mb", "oa_arbase_mb", "arbase_mgn",
    }
    if model_type_str in oa_names or model_type_str.startswith("oa_") or model_type_str.startswith("openanimals_"):
        from .openanimals_models import OpenAnimalsVideoModel
        return OpenAnimalsVideoModel(
            model_name=model_type_str,
            pooling_type=getattr(cfg, "pooling_type", "attention"),
            num_classes=getattr(cfg, "num_classes", 0),
            chunk_size=getattr(cfg, "chunk_size", 32),
            unfreeze_blocks=getattr(cfg, "unfreeze_blocks", 2),
            freeze_epochs=getattr(cfg, "freeze_epochs", None),
        )

    # TF-CLIP (Text-Free CLIP Video Re-ID)
    if model_type_str in ("tfclip", "tf_clip", "tf-clip", "clip"):
        from .tfclip import TFCLIPVideoModel
        return TFCLIPVideoModel(cfg)

    # --- DINOv2 / CLIP BoT Video Re-ID Baseline ---
    if getattr(cfg, "reid_method", None) == "bot" or model_type_str in ("dinov2", "baseline", "dinov2_baseline", "clip_baseline", "baseline_clip"):
        return VideoReID(cfg)

    else:
        # Fallback for unsupported or misspelled model configurations
        raise ValueError(f"Unknown model architecture requested: {model_type}")
