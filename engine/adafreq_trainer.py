import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

from models.adafreq import TorchHighFrequencyFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAFREQ_ROOT = PROJECT_ROOT / "AdaFreq"


def _load_adafreq_training_components():
    """Import AdaFreq solver and loss modules without namespace collisions."""
    if str(ADAFREQ_ROOT) not in sys.path:
        sys.path.insert(0, str(ADAFREQ_ROOT))

    saved_modules = {}
    for mod_name in (
        "solver",
        "solver.make_optimizer",
        "solver.scheduler_factory",
        "solver.cosine_lr",
        "solver.scheduler",
        "loss",
        "loss.triplet_loss",
        "loss.center_loss",
    ):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    try:
        from solver.make_optimizer import make_optimizer
        from solver.scheduler_factory import create_scheduler
        from loss.triplet_loss import TripletLoss
        from loss.center_loss import CenterLoss
        return make_optimizer, create_scheduler, TripletLoss, CenterLoss
    finally:
        for k, v in saved_modules.items():
            sys.modules[k] = v


_make_optimizer, _create_scheduler, _TripletLoss, _CenterLoss = _load_adafreq_training_components()


@torch.no_grad()
def evaluate_adafreq(model, query_loader, gallery_loader, device):
    """
    Evaluate Rank-1, Rank-5, Rank-10, and mAP using only the original RGB features
    (`bottleneck(global_feat)` with L2 normalization), matching the paper.
    """
    model.eval()

    def _extract(loader):
        feats, dogs, vids = [], [], []
        for videos, _labels, dog_ids, video_ids in loader:
            out = model(videos.to(device, non_blocking=True))
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = F.normalize(out.float(), p=2, dim=1)
            feats.append(out.cpu())
            dogs.extend([str(d) for d in dog_ids])
            vids.extend([str(v) for v in video_ids])
        return torch.cat(feats, dim=0), np.array(dogs), np.array(vids)

    q_feat, q_dogs, q_vids = _extract(query_loader)
    g_feat, g_dogs, g_vids = _extract(gallery_loader)

    dist_mat = torch.cdist(q_feat.to(device), g_feat.to(device), p=2).cpu().numpy()

    num_q = len(q_dogs)
    ranks = np.zeros(num_q, dtype=np.int32)
    aps = []
    valid_q = 0

    for i in range(num_q):
        q_dog, q_vid = q_dogs[i], q_vids[i]
        dists = dist_mat[i].copy()
        same_video = (g_dogs == q_dog) & (g_vids == q_vid)
        dists[same_video] = np.inf

        order = np.argsort(dists)
        matches = (g_dogs[order] == q_dog) & (~same_video[order])
        if not np.any(matches):
            continue

        valid_q += 1
        ranks[valid_q - 1] = int(np.argmax(matches))
        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches[matches] / (np.where(matches)[0] + 1.0)
        aps.append(float(np.mean(precision_at_k)))

    if valid_q == 0:
        return 0.0, 0.0, 0.0, 0.0

    ranks = ranks[:valid_q]
    return (
        float(np.mean(ranks < 1)),
        float(np.mean(ranks < 5)),
        float(np.mean(ranks < 10)),
        float(np.mean(aps)),
    )


def train_adafreq(cfg, model, train_loader, query_loader, gallery_loader):
    """
    Train AdaFreqVideoModel using the exact optimization, loss (ID + SoftTriplet + SmoothL1),
    HighFrequencyFilter (D=70, CutMix), and CosineLRScheduler from the AdaFreq paper & repo.
    """
    device = cfg.device
    model.to(device)
    af_cfg = model.af_cfg

    dummy_center = _CenterLoss(num_classes=max(1, model.num_classes), feat_dim=model.in_planes, use_gpu=(device.type == "cuda"))
    optimizer, _ = _make_optimizer(af_cfg, model, dummy_center)
    scheduler = _create_scheduler(af_cfg, optimizer)

    # Exact AdaFreq loss components (NO_MARGIN=True -> SoftTriplet; IF_LABELSMOOTH='off' -> CrossEntropy)
    triplet_fn = _TripletLoss(None if af_cfg.MODEL.NO_MARGIN else af_cfg.SOLVER.MARGIN)
    l1_loss_fn = nn.SmoothL1Loss(reduction="mean")
    high_freq_filter = TorchHighFrequencyFilter(
        D=70.0,
        pixel_mean=af_cfg.INPUT.PIXEL_MEAN,
        pixel_std=af_cfg.INPUT.PIXEL_STD,
    ).to(device)

    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    max_epochs = int(af_cfg.SOLVER.MAX_EPOCHS)
    eval_period = int(getattr(cfg, "eval_period", 5))
    lam_weight = float(getattr(cfg, "adafreq_lambda", 0.1))
    start_epoch = 1
    best_mAP = -1.0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_config()

    # Optional resume support
    latest_ckpt = cfg.output_dir / "latest_model.pth"
    if getattr(cfg, "resume", False) and latest_ckpt.exists():
        print(f"[AdaFreq] Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mAP = float(ckpt.get("best_mAP", ckpt.get("mAP", -1.0)))
        print(f"[AdaFreq] Resumed at epoch {start_epoch} (best_mAP={best_mAP*100:.2f}%)")

    print("\n" + "=" * 68)
    print("STARTING ECCV-24 ADAFREQ VIDEO TRAINING (Exact Paper Configuration)")
    print(f"  Backbone         : {af_cfg.MODEL.TRANSFORMER_TYPE} (stride={list(af_cfg.MODEL.STRIDE_SIZE)}, mu={model.mu} -> keep {model.num_keep} patches)")
    print(f"  Temporal Pooling : {model.pooling_type}")
    print(f"  Input Resolution : {cfg.img_size[0]}x{cfg.img_size[1]} ({cfg.clip_len} frames/clip)")
    print(f"  Augmentations    : Rot(15 deg), Brightness(p=0.5), Contrast(p=0.5), Pad(10)")
    print(f"  Batch Size       : {cfg.batch_size} (P={cfg.batch_size // cfg.k}, K={cfg.k})")
    print(f"  Optimizer        : {af_cfg.SOLVER.OPTIMIZER_NAME} (base_lr={af_cfg.SOLVER.BASE_LR}, momentum={af_cfg.SOLVER.MOMENTUM}, wd={af_cfg.SOLVER.WEIGHT_DECAY})")
    print(f"  Scheduler        : CosineLRScheduler ({max_epochs} epochs, {af_cfg.SOLVER.WARMUP_EPOCHS} warmup epochs)")
    print(f"  Losses           : ID ((1-lam)*CE_orig + lam*CE_high, lam={lam_weight}) + Triplet (0.7*Tri_orig + 0.2*Tri_high + 0.1*SmoothL1_patch)")
    print(f"  Output Directory : {cfg.output_dir}")
    print("=" * 68 + "\n")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        scheduler.step(epoch)
        model.train()

        total_loss, total_id, total_tri, total_l1, total_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for videos, pids, _dog_ids, _video_ids in tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{max_epochs:03d}", leave=False
        ):
            videos = videos.to(device, non_blocking=True)
            target_pids = pids.to(device, dtype=torch.long, non_blocking=True)

            # Generate High-Frequency filtered video frames on GPU
            B, T, C, H, W = videos.shape
            high_freq_flat = high_freq_filter.apply_filter(videos.reshape(B * T, C, H, W))
            high_freq_videos = high_freq_flat.reshape(B, T, C, H, W)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                scores, feats = model(videos, y=high_freq_videos, label=target_pids)
                cls_score_0, cls_score_1 = scores[0], scores[1]
                global_feat, high_feat, ori_part, high_feat_1 = (
                    feats[0].float(),
                    feats[1].float(),
                    feats[2].float(),
                    feats[3].float(),
                )

                # 1. ID Loss: 0.9 * CE(score[0]) + 0.1 * CE(score[1])
                loss_id = (1.0 - lam_weight) * F.cross_entropy(cls_score_0, target_pids) + lam_weight * F.cross_entropy(cls_score_1, target_pids)

                # 2. Patch Alignment SmoothL1 Loss across selected top-mu patches (excluding CLS token at index 0)
                # In AdaFreq/loss/make_loss.py:66-74, L1_LOSS is summed across the batch of samples (mean over patches per sample)
                feat_1 = ori_part[:, 1:]      # (B*T, num_keep, 768)
                feat_2 = high_feat_1[:, 1:]   # (B*T, num_keep, 768)
                # Average over T frames per clip so video batch scale matches image batch scale B
                per_sample_l1 = F.smooth_l1_loss(feat_1, feat_2, reduction="none").mean(dim=(1, 2))  # (B*T,)
                loss_l1 = per_sample_l1.view(B, T).mean(dim=1).sum()

                # 3. Triplet + Patch Loss: 0.7 * Tri(global_feat) + 0.2 * Tri(high_feat) + 0.1 * L1_LOSS
                tri_orig = triplet_fn(global_feat, target_pids)[0]
                tri_high = triplet_fn(high_feat, target_pids)[0]
                loss_tri = 0.7 * tri_orig + 0.2 * tri_high + lam_weight * loss_l1

                loss = (
                    af_cfg.MODEL.ID_LOSS_WEIGHT * loss_id
                    + af_cfg.MODEL.TRIPLET_LOSS_WEIGHT * loss_tri
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = (cls_score_0.max(1)[1] == target_pids).float().mean().item()
            total_loss += float(loss.detach().item())
            total_id += float(loss_id.detach().item())
            total_tri += float((0.7 * tri_orig + 0.2 * tri_high).detach().item())
            total_l1 += float((lam_weight * loss_l1).detach().item())
            total_acc += acc
            n_batches += 1

        dt = time.time() - t0
        cur_lr = scheduler._get_lr(epoch)[0]
        nb = max(1, n_batches)

        if epoch % eval_period == 0 or epoch == max_epochs:
            r1, r5, r10, mAP = evaluate_adafreq(model, query_loader, gallery_loader, device)
            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP

            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "mAP": mAP,
                "best_mAP": best_mAP,
                "rank1": r1,
                "rank5": r5,
                "rank10": r10,
            }
            torch.save(payload, cfg.output_dir / "latest_model.pth")
            if is_best:
                torch.save(payload, cfg.output_dir / "best_model.pth")

            best_str = " [BEST]" if is_best else ""
            print(
                f"Epoch [{epoch:03d}/{max_epochs:03d}] ({dt:.1f}s) | "
                f"Loss: {total_loss/nb:.3f} (ID: {total_id/nb:.3f}, Tri: {total_tri/nb:.3f}, PatchL1: {total_l1/nb:.3f}) | "
                f"Acc: {total_acc/nb*100:.1f}% | LR: {cur_lr:.2e} | "
                f"R1: {r1*100:.2f}% | R5: {r5*100:.2f}% | R10: {r10*100:.2f}% | mAP: {mAP*100:.2f}%{best_str}"
            )
        else:
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_mAP": best_mAP,
            }
            torch.save(payload, cfg.output_dir / "latest_model.pth")
            print(
                f"Epoch [{epoch:03d}/{max_epochs:03d}] ({dt:.1f}s) | "
                f"Loss: {total_loss/nb:.3f} (ID: {total_id/nb:.3f}, Tri: {total_tri/nb:.3f}, PatchL1: {total_l1/nb:.3f}) | "
                f"Acc: {total_acc/nb*100:.1f}% | LR: {cur_lr:.2e}"
            )

    print(f"\n[AdaFreq] Training complete! Best mAP: {best_mAP*100:.2f}%")
    print(f"[AdaFreq] Checkpoints saved to: {cfg.output_dir}")
