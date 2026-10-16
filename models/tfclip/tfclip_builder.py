from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .clip_model import load_clip_vitb16
from .modules import Temporal_Memory_Difusion, ImageSpecificPrompt, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Batch-hard Euclidean triplet loss with specified margin."""
    embeddings = embeddings.float()
    dist_mat = torch.cdist(embeddings, embeddings, p=2)
    N = embeddings.size(0)
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = dist_mat.masked_fill(~is_pos, float("-inf")).max(dim=1)[0]
    dist_an = dist_mat.masked_fill(~is_neg, float("inf")).min(dim=1)[0]

    y = torch.ones_like(dist_an)
    loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)
    return loss


class TFCLIPVideoModel(nn.Module):
    """
    TF-CLIP: Learning Text-Free CLIP for Video-Based Re-identification (AAAI 2024).
    Adapted for DogReID-1553 video re-identification.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.is_tfclip = True
        self.is_openanimals = False

        img_h, img_w = getattr(cfg, "img_size", (224, 224))
        stride = 16
        h_resolution = (img_h - 16) // stride + 1
        w_resolution = (img_w - 16) // stride + 1

        self.num_classes = getattr(cfg, "num_classes", 744)
        self.clip_len = getattr(cfg, "clip_len", 8)
        self.neck_feat = getattr(cfg, "neck_feat", "before")

        # 1. Visual Backbone: OpenAI CLIP ViT-B/16
        self.image_encoder = load_clip_vitb16(
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            patch_size=16,
            stride_size=stride
        )

        # 2. Temporal Memory Diffusion (TMD) across T frames
        self.TMD = Temporal_Memory_Difusion(
            width=768,
            layers=1,
            heads=12,
            droppath=None,
            T=self.clip_len
        )

        # 3. Sequence-Specific Prompt (SSP) for Text-Free CLIP
        self.SSP = ImageSpecificPrompt(layers=2, embed_dim=512)

        # 4. Bottlenecks (BatchNorm1d)
        self.bottleneck = nn.BatchNorm1d(768)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(512)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.bottleneck_proj_temp = nn.BatchNorm1d(768)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)

        self.bottleneck_proj_temp2 = nn.BatchNorm1d(768)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        # 5. Classifiers
        self.classifier2 = nn.Linear(768, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)

        self.classifier_proj = nn.Linear(512, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.classifier_proj_temp = nn.Linear(768, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)

        # 6. CLIP-Memory Bank: Precomputed centroids [num_classes, 512]
        self.register_buffer("clip_memory", torch.zeros(self.num_classes, 512))
        self.register_buffer("memory_initialized", torch.tensor(False, dtype=torch.bool))

        # Loss weights
        self.id_loss_weight = getattr(cfg, "id_loss_weight", 0.25)
        self.triplet_loss_weight = getattr(cfg, "triplet_loss_weight", 1.0)
        self.i2t_loss_weight = getattr(cfg, "i2t_loss_weight", 1.0)
        self.margin = getattr(cfg, "margin", 0.3)
        self.label_smoothing = 0.1

    def _build_stage1_loader(self, train_loader):
        """
        Build a deterministic, unaugmented DataLoader over all training videos
        (no RandomErasing, no RandomCrop, no MPerClassSampler, drop_last=False)
        matching TF-CLIP's train_loader_stage1 / train_set_normal.
        """
        from torch.utils.data import DataLoader, Subset
        from data.dataset import DOGVideoREIDDataset
        from data.reid_transforms import build_video_transforms

        eval_tf = build_video_transforms(self.cfg, is_train=False)
        train_ds = train_loader.dataset
        base_ds = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

        clean_base_ds = DOGVideoREIDDataset(
            root_dir=self.cfg.data_root,
            split_file=self.cfg.split_file,
            split="train",
            is_train=False,
            clip_len=self.cfg.clip_len,
            transform=eval_tf,
            world=self.cfg.world,
            label_map=getattr(base_ds, "id_map", None),
            mask_dog=getattr(self.cfg, "mask_dog", False),
            force_yolo=getattr(self.cfg, "force_yolo", False),
            bbox_file=getattr(self.cfg, "bbox_file", None),
        )
        stage1_ds = Subset(clean_base_ds, train_ds.indices) if isinstance(train_ds, Subset) else clean_base_ds

        num_workers = getattr(self.cfg, "num_workers", 4)
        loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
        if num_workers > 0:
            loader_kwargs.update(persistent_workers=False, prefetch_factor=4, timeout=300)

        return DataLoader(
            stage1_ds,
            batch_size=min(getattr(self.cfg, "batch_size", 16), 32),
            shuffle=False,
            drop_last=False,
            **loader_kwargs
        )

    @torch.no_grad()
    def compute_clip_memory(self, train_loader, device):
        """
        Stage 1: Precompute identity centroid representations (CLIP-Memory)
        using the frozen CLIP visual backbone across the full unaugmented training set.
        """
        was_training = self.training
        self.eval()
        device = torch.device(device)

        # Check disk cache first to avoid recomputing Stage 1
        cache_dir = getattr(self.cfg, "output_dir", None)
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"clip_memory_{self.num_classes}ids.pt"
            if cache_file.exists():
                cached = torch.load(cache_file, map_location=device)
                unobserved_count = int((cached.norm(dim=-1) < 1.0).sum().item())
                if cached.shape == (self.num_classes, 512) and unobserved_count == 0:
                    print(f"=> [TF-CLIP] Found valid cached CLIP-Memory at {cache_file}. Loading...")
                    self.clip_memory.copy_(cached)
                    self.memory_initialized.copy_(torch.tensor(True, dtype=torch.bool))
                    self.train(was_training)
                    return
                else:
                    print(f"=> [TF-CLIP] Cached CLIP-Memory at {cache_file} has {unobserved_count} unobserved IDs; recomputing cleanly...")

        stage1_loader = self._build_stage1_loader(train_loader)
        print(f"=> [TF-CLIP] Generating CLIP-Memory identity centroids over {len(stage1_loader.dataset)} clean training videos...")
        feats_by_id = {}

        for batch in tqdm(stage1_loader, desc="[TF-CLIP] Building Memory"):
            clips = batch[0]  # [B, T, C, H, W]
            labels = batch[1]  # [B]

            B, T, C, H, W = clips.shape
            clips_flat = clips.view(-1, C, H, W).to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                _, image_features_proj_raw = self.image_encoder(clips_flat)
                # Take CLS token projection [B*T, 512] -> [B, T, 512] -> mean across T: [B, 512]
                proj_cls = image_features_proj_raw[:, 0].view(B, T, -1).mean(dim=1)

            for label, feat in zip(labels, proj_cls):
                lbl = int(label.item()) if hasattr(label, "item") else int(label)
                if lbl not in feats_by_id:
                    feats_by_id[lbl] = []
                feats_by_id[lbl].append(feat.detach().float().cpu())

        # Construct memory matrix [num_classes, 512]
        memory = torch.zeros(self.num_classes, 512, device=device)
        missing_ids = 0
        for lbl in range(self.num_classes):
            if lbl in feats_by_id and len(feats_by_id[lbl]) > 0:
                stacked = torch.stack(feats_by_id[lbl], dim=0)
                centroid = stacked.mean(dim=0).to(device)
                memory[lbl] = centroid
            else:
                missing_ids += 1
                memory[lbl] = torch.randn(512, device=device) * 0.01

        self.clip_memory.copy_(memory)
        self.memory_initialized.copy_(torch.tensor(True, dtype=torch.bool))
        print(f"=> [TF-CLIP] Successfully initialized CLIP-Memory for {self.num_classes - missing_ids}/{self.num_classes} identities.")

        # Save to disk cache
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"clip_memory_{self.num_classes}ids.pt"
            torch.save(memory.cpu(), cache_file)
            print(f"=> [TF-CLIP] Saved CLIP-Memory cache to {cache_file}.")

        self.train(was_training)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            x: Video tensor of shape [B, T, C, H, W]
            targets: Optional ground-truth identity labels of shape [B]
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(-1, C, H, W)

        # 1. Forward through CLIP ViT-B/16 (chunked by chunk_size to prevent CUDA OOM)
        chunk_size = getattr(self.cfg, "chunk_size", 32)
        total_frames = x_flat.size(0)
        if total_frames > chunk_size:
            feats_list = []
            proj_list = []
            for i in range(0, total_frames, chunk_size):
                chunk = x_flat[i : i + chunk_size]
                f_chunk, p_chunk = self.image_encoder(chunk)
                feats_list.append(f_chunk)
                proj_list.append(p_chunk)
            image_features = torch.cat(feats_list, dim=0)
            image_features_proj_raw = torch.cat(proj_list, dim=0)
        else:
            image_features, image_features_proj_raw = self.image_encoder(x_flat)

        # 2. Extract global frame representations (CLS token)
        img_feature = image_features[:, 0].view(B, T, -1).mean(1)       # [B, 768]
        img_feature_proj = image_features_proj_raw[:, 0].view(B, T, -1).mean(1)  # [B, 512]

        # 3. Temporal Memory Diffusion (TMD) branch
        ft_detached = image_features.detach()
        f_sp = self.TMD(ft_detached.permute(1, 0, 2))  # [L+1, B*T, 768]
        f_sp2 = f_sp.permute(1, 0, 2)                  # [B*T, L+1, 768]
        cls_f_sp = f_sp2.mean(1)                       # [B*T, 768] (frame-level)
        cls_f_tp = cls_f_sp.view(B, T, -1).mean(1)    # [B, 768] (temporal-diffused clip)

        # 4. Bottleneck normalization
        feat = self.bottleneck(img_feature)                    # [B, 768]
        feat_proj = self.bottleneck_proj(img_feature_proj)     # [B, 512]
        feat_proj_frame = self.bottleneck_proj_temp(cls_f_sp)  # [B*T, 768]
        feat_proj_temp = self.bottleneck_proj_temp2(cls_f_tp)  # [B, 768]

        # 5. Training vs Inference Branch
        if self.training:
            # Memory adaptation via Sequence-Specific Prompt (SSP)
            text_memory = self.clip_memory.unsqueeze(0).expand(B, -1, -1)  # [B, K, 512]
            vid_proj = image_features_proj_raw.view(B, T, -1, 512).mean(1)  # [B, L, 512]
            text_adapted = text_memory + self.SSP(text_memory, vid_proj)   # [B, K, 512]

            # Sequence-to-Memory alignment logits
            logits = torch.einsum("bd,bkd->bk", img_feature_proj, text_adapted)  # [B, K]

            # Class prediction scores
            cls_score = self.classifier2(feat)                             # [B, K]
            cls_score_proj = self.classifier_proj(feat_proj)               # [B, K]
            cls_score_proj_temp = self.classifier_proj_temp(feat_proj_temp)  # [B, K]
            cls_score_proj_frame = self.classifier_proj_temp(feat_proj_frame) # [B*T, K]

            scores = [cls_score, cls_score_proj, cls_score_proj_temp, cls_score_proj_frame]
            feats = [img_feature, img_feature_proj, cls_f_tp]

            return {
                "scores": scores,
                "feats": feats,
                "logits": logits,
                "T": T
            }
        else:
            # Inference: return normalized embeddings
            if self.neck_feat == "after":
                out_feat = torch.cat([feat, feat_proj], dim=1)  # [B, 1280]
            else:
                out_feat = torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)  # [B, 2048]
            return F.normalize(out_feat, p=2, dim=1)

    def compute_loss(self, outputs: dict, targets: torch.Tensor):
        """
        Computes the joint multi-task loss defined in TF-CLIP:
        L = L_id + L_tri + L_i2t + L_frame
        """
        scores = [s.float() for s in outputs["scores"]]
        feats = [f.float() for f in outputs["feats"]]
        logits = outputs["logits"].float()
        T = outputs["T"]
        B = targets.size(0)

        # 1. Identity Cross-Entropy with Label Smoothing (0.1) on global features
        score_global = scores[:3]
        loss_id = sum(
            F.cross_entropy(s, targets, label_smoothing=self.label_smoothing) for s in score_global
        ) * self.id_loss_weight

        # 2. Batch-Hard Triplet Ranking Loss on visual features
        loss_triplet = sum(
            batch_hard_triplet_loss(f, targets, margin=self.margin) for f in feats
        ) * self.triplet_loss_weight

        # 3. Contrastive Alignment Loss (Image-to-Prompt)
        loss_i2t = F.cross_entropy(
            logits, targets, label_smoothing=self.label_smoothing
        ) * self.i2t_loss_weight

        # 4. Frame-Level Classification Loss
        score_frame = scores[3]  # [B*T, K]
        targets_expanded = targets.unsqueeze(1).expand(B, T).contiguous().view(B * T)
        loss_frame = F.cross_entropy(
            score_frame, targets_expanded, label_smoothing=self.label_smoothing
        ) / T

        total_loss = loss_id + loss_triplet + loss_i2t + loss_frame
        return total_loss, loss_triplet, loss_id

    def get_parameter_groups(self, base_lr: float = 5e-6, weight_decay: float = 1e-4, bias_lr_factor: float = 2.0, large_fc_lr: bool = True):
        """
        Matches TF-CLIP solver/make_optimizer_prompt.py (make_optimizer_2stage):
        - Base parameters (CLIP backbone, TMD, SSP, bottlenecks, classifier2, classifier_proj): base_lr (5e-6)
        - Bias parameters: base_lr * bias_lr_factor (2x = 1e-5)
        - classifier_proj_temp (when large_fc_lr=True): base_lr * 10 (5e-5)
        """
        param_groups = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            lr = base_lr
            wd = weight_decay
            if "bias" in key:
                lr = base_lr * bias_lr_factor
            if large_fc_lr and ("classifier_proj_temp" in key or "arcface" in key):
                lr = lr * 10.0
            param_groups.append({
                "params": [value],
                "lr": lr,
                "weight_decay": wd,
                "is_backbone": key.startswith("image_encoder.")
            })
        return param_groups
