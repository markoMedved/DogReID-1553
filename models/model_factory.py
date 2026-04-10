from .video_resnet_reid import VideoResNetReID


def build_model(cfg):

    if cfg.model == "resnet":

        model = VideoResNetReID(
            backbone=cfg.backbone,
            embedding_dim=cfg.embedding_dim
        )

        return model

    raise ValueError("Unknown model")