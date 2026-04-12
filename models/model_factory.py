# from .video_resnet_reid import VideoResNetReID
from .vit_builder import VideoViT

def build_model(cfg):

    if cfg.model == "vit":

        model = VideoViT(embedding_dim=cfg.embedding_dim)

    else:

        raise ValueError("Unknown model")


    return model