from .video_resnet_reid import VideoResNetReID
from models.video_vit_reid import VideoViTReID

def build_model(cfg):

    if cfg.model == "resnet":

        model = VideoResNetReID(
            backbone=cfg.backbone,
            embedding_dim=cfg.embedding_dim
        )
    elif cfg.model == "vit":
        model = VideoViTReID(
            embedding_dim=cfg.embedding_dim
        )

        return model

    raise ValueError("Unknown model")