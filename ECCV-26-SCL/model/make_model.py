import torch
import torch.nn as nn

from .modules.freq_norm_vit import SCNorm1D
from .vit import vit_base_patch16_224_TransReID


def weights_init_kaiming(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        if module.affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def weights_init_classifier(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def build_norm(num_features, dist_train):
    if dist_train:
        return nn.SyncBatchNorm(num_features)
    return nn.BatchNorm1d(num_features)


class MetaNModel(nn.Module):
    def __init__(self, num_classes, species_num, cfg):
        super().__init__()
        if cfg.MODEL.TRANSFORMER_TYPE != "vit_base_patch16_224_TransReID":
            raise ValueError(
                "This clean MetaN release only supports "
                "vit_base_patch16_224_TransReID."
            )

        self.image_encoder = vit_base_patch16_224_TransReID(cfg)
        self.in_planes = getattr(self.image_encoder, "embed_dim", 768)
        self.num_classes = num_classes
        self.neck_feat = cfg.TEST.NECK_FEAT

        self.scnorm_before_bn = None
        if cfg.CHANGE.METHODS.SCNORM_BEFORE_BNNECK:
            self.scnorm_before_bn = SCNorm1D(
                self.in_planes,
                base=cfg.CHANGE.METHODS.SCNORM_BNNECK_BASE,
                Ts=1e-1,
            )

        self.bottleneck = build_norm(self.in_planes, cfg.MODEL.DIST_TRAIN)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        if cfg.MODEL.PRETRAIN_CHOICE == "imagenet":
            self.image_encoder.load_param(cfg.MODEL.PRETRAIN_PATH)
            print(f"Loaded ImageNet pretrained weights from {cfg.MODEL.PRETRAIN_PATH}")

    def forward(self, forward_type="main", x=None, label=None, spe_label=None, **kwargs):
        if forward_type != "main":
            raise ValueError("The clean MetaN model only supports forward_type='main'.")
        if x is None:
            raise ValueError("Input tensor x is required.")

        features = self.image_encoder(forward_type="main", x=x)
        global_feat = features[:, 0]
        extra_info = {}

        if self.scnorm_before_bn is not None:
            global_feat = self.scnorm_before_bn(global_feat)
            extra_info["semantic_feat"] = global_feat

        feat = self.bottleneck(global_feat)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, extra_info

        if self.neck_feat == "after":
            return feat, feat
        return global_feat, feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location="cpu")
        if "model" in param_dict:
            param_dict = param_dict["model"]
        if "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        model_state = self.state_dict()
        copied = 0
        for key, value in param_dict.items():
            clean_key = key.replace("module.", "")
            if clean_key in model_state and model_state[clean_key].shape == value.shape:
                model_state[clean_key].copy_(value)
                copied += 1
        print(f"Loaded {copied} tensors from {trained_path}")


def make_model(cfg, num_class, species_num):
    print("=========== building MetaN model ===========")
    return MetaNModel(num_class, species_num, cfg)
