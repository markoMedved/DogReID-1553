import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCL_ROOT = PROJECT_ROOT / "ECCV-26-SCL"


def _load_scl_training_components():
    """Import ECCV-26-SCL solver and loss modules without namespace collisions."""
    if str(SCL_ROOT) not in sys.path:
        sys.path.insert(0, str(SCL_ROOT))

    saved_modules = {}
    for mod_name in (
        "solver",
        "solver.make_optimizer",
        "solver.scheduler_factory",
        "solver.cosine_lr",
        "solver.scheduler",
        "loss",
        "loss.triplet_loss",
        "loss.nfc",
    ):
        if mod_name in sys.modules:
            saved_modules[mod_name] = sys.modules.pop(mod_name)

    try:
        from solver.make_optimizer import make_optimizer
        from solver.scheduler_factory import create_scheduler
        from loss.triplet_loss import TripletLoss
        from loss.nfc import EMAModel, NFCTrainer
        return make_optimizer, create_scheduler, TripletLoss, EMAModel, NFCTrainer
    finally:
        for k, v in saved_modules.items():
            sys.modules[k] = v


_make_optimizer, _create_scheduler, _TripletLoss, _EMAModel, _NFCTrainer = _load_scl_training_components()


@torch.no_grad()
def evaluate_scl(model, query_loader, gallery_loader, device):
    """Evaluate Rank-1, Rank-5, Rank-10, and mAP using L2-normalized features."""
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


def train_scl(cfg, model, train_loader, query_loader, gallery_loader):
    """
    Train SCLVideoModel using the exact optimization, loss (ID + Triplet + NFC),
    EMA teacher, and CosineLRScheduler from ECCV-26-SCL/processor/processor.py.
    """
    device = cfg.device
    model.to(device)
    scl_cfg = model.scl_cfg

    optimizer = _make_optimizer(scl_cfg, model)
    scheduler = _create_scheduler(scl_cfg, optimizer)

    id_loss_fn = F.cross_entropy
    tri_loss_fn = _TripletLoss(margin=scl_cfg.SOLVER.MARGIN)
    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    nfc_trainer = None
    ema_model = None
    if scl_cfg.CHANGE.METHODS.NFC_TRAINING:
        nfc_trainer = _NFCTrainer(
            queue_size=scl_cfg.CHANGE.METHODS.NFC_QUEUE_SIZE,
            k1=scl_cfg.CHANGE.METHODS.NFC_K1,
            k2=scl_cfg.CHANGE.METHODS.NFC_K2,
            temperature=scl_cfg.CHANGE.METHODS.NFC_TEMPERATURE,
            feature_dim=scl_cfg.CHANGE.METHODS.NFC_FEATURE_DIM,
            same_species_weight=scl_cfg.CHANGE.METHODS.NFC_SAME_SPECIES_WEIGHT,
            enable_cross_species=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES,
            cross_species_k1=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_K1,
            cross_species_k2=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_K2,
            cross_species_weight=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_WEIGHT,
            cross_species_margin=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_MARGIN,
            cross_species_metric=scl_cfg.CHANGE.METHODS.NFC_CROSS_SPECIES_METRIC,
        )
        ema_model = _EMAModel(model, momentum=scl_cfg.CHANGE.METHODS.NFC_EMA_MOMENTUM)

    max_epochs = int(scl_cfg.SOLVER.MAX_EPOCHS)
    eval_period = int(getattr(cfg, "eval_period", 5))
    start_epoch = 1
    best_mAP = -1.0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_config()

    # Optional resume support
    latest_ckpt = cfg.output_dir / "latest_model.pth"
    if getattr(cfg, "resume", False) and latest_ckpt.exists():
        print(f"[SCL] Resuming from checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema_model is not None and "ema_shadow" in ckpt:
            for k_s, v_s in ckpt["ema_shadow"].items():
                if k_s in ema_model.shadow:
                    ema_model.shadow[k_s] = v_s.to(device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_mAP = float(ckpt.get("best_mAP", ckpt.get("mAP", -1.0)))
        print(f"[SCL] Resumed at epoch {start_epoch} (best_mAP={best_mAP*100:.2f}%)")

    print("\n" + "=" * 65)
    print("STARTING ECCV-26-SCL (MetaN) VIDEO TRAINING")
    print(f"  Backbone         : {scl_cfg.MODEL.TRANSFORMER_TYPE} + SCFG at {list(scl_cfg.CHANGE.METHODS.FREQ_NORM_POSITIONS)}")
    print(f"  Temporal Pooling : {model.pooling_type}")
    print(f"  Input Resolution : {cfg.img_size[0]}x{cfg.img_size[1]} ({cfg.clip_len} frames/clip)")
    print(f"  Batch Size       : {cfg.batch_size} (P={cfg.batch_size // cfg.k}, K={cfg.k})")
    print(f"  Optimizer        : {scl_cfg.SOLVER.OPTIMIZER_NAME} (base_lr={scl_cfg.SOLVER.BASE_LR}, momentum={scl_cfg.SOLVER.MOMENTUM}, wd={scl_cfg.SOLVER.WEIGHT_DECAY})")
    print(f"  Scheduler        : CosineLRScheduler ({max_epochs} epochs, {scl_cfg.SOLVER.WARMUP_EPOCHS} warmup epochs)")
    print(f"  Losses           : ID (w={scl_cfg.MODEL.ID_LOSS_WEIGHT}) + Triplet(m={scl_cfg.SOLVER.MARGIN}, w={scl_cfg.MODEL.TRIPLET_LOSS_WEIGHT}) + NFC(w={scl_cfg.CHANGE.METHODS.NFC_LOSS_WEIGHT}, warmup={scl_cfg.CHANGE.METHODS.NFC_WARMUP_EPOCHS} ep)")
    print(f"  Output Directory : {cfg.output_dir}")
    print("=" * 65 + "\n")

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        model.set_freq_norm_epoch(epoch)
        scheduler.step(epoch)
        model.train()

        total_loss, total_id, total_tri, total_nfc, total_acc = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for videos, pids, _dog_ids, _video_ids in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}/{max_epochs:02d}", leave=False
        ):
            videos = videos.to(device, non_blocking=True)
            target_pids = pids.to(device, dtype=torch.long, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                score, feat = model(videos)
                loss_id = id_loss_fn(score, target_pids)
                loss_tri = tri_loss_fn(feat.float(), target_pids)[0]
                loss = (
                    loss_id * scl_cfg.MODEL.ID_LOSS_WEIGHT
                    + loss_tri * scl_cfg.MODEL.TRIPLET_LOSS_WEIGHT
                )

                loss_nfc = torch.zeros((), device=device)
                if (
                    nfc_trainer is not None
                    and ema_model is not None
                    and epoch > scl_cfg.CHANGE.METHODS.NFC_WARMUP_EPOCHS
                ):
                    ema_model.apply_shadow()
                    with torch.no_grad():
                        _, teacher_feat = model(videos)
                    ema_model.restore()

                    loss_nfc, _ = nfc_trainer.compute_nfc_loss(
                        feat.float(),
                        teacher_feat.float(),
                        target_pids,
                        device,
                        spes=None,
                    )
                    nfc_trainer.update_queue(teacher_feat.float(), target_pids, spes=None)
                    loss = loss + scl_cfg.CHANGE.METHODS.NFC_LOSS_WEIGHT * loss_nfc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                ema_model.update()

            acc = (score.max(1)[1] == target_pids).float().mean().item()
            total_loss += float(loss.detach().item())
            total_id += float(loss_id.detach().item())
            total_tri += float(loss_tri.detach().item())
            total_nfc += float(loss_nfc.detach().item())
            total_acc += acc
            n_batches += 1

        dt = time.time() - t0
        cur_lr = scheduler._get_lr(epoch)[0]
        nb = max(1, n_batches)

        if epoch % eval_period == 0 or epoch == max_epochs:
            r1, r5, r10, mAP = evaluate_scl(model, query_loader, gallery_loader, device)
            is_best = mAP > best_mAP
            if is_best:
                best_mAP = mAP

            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema_shadow": {k_s: v_s.cpu() for k_s, v_s in ema_model.shadow.items()} if ema_model is not None else {},
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
                f"Epoch [{epoch:02d}/{max_epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {total_loss/nb:.3f} (ID: {total_id/nb:.3f}, Tri: {total_tri/nb:.3f}, NFC: {total_nfc/nb:.3f}) | "
                f"Acc: {total_acc/nb*100:.1f}% | LR: {cur_lr:.2e} | "
                f"R1: {r1*100:.2f}% | R5: {r5*100:.2f}% | R10: {r10*100:.2f}% | mAP: {mAP*100:.2f}%{best_str}"
            )
        else:
            # Always update latest_model.pth for seamless preemption resume
            payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema_shadow": {k_s: v_s.cpu() for k_s, v_s in ema_model.shadow.items()} if ema_model is not None else {},
                "epoch": epoch,
                "best_mAP": best_mAP,
            }
            torch.save(payload, cfg.output_dir / "latest_model.pth")
            print(
                f"Epoch [{epoch:02d}/{max_epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {total_loss/nb:.3f} (ID: {total_id/nb:.3f}, Tri: {total_tri/nb:.3f}, NFC: {total_nfc/nb:.3f}) | "
                f"Acc: {total_acc/nb*100:.1f}% | LR: {cur_lr:.2e}"
            )

    print(f"\n[SCL] Training complete! Best mAP: {best_mAP*100:.2f}%")
    print(f"[SCL] Checkpoints saved to: {cfg.output_dir}")
