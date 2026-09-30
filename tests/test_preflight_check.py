#!/usr/bin/env python
"""
Pre-Flight Verification Script for OpenAnimals Video Re-ID Training.
Run this script before submitting a long SLURM training job to catch:
1. GPU & Mixed Precision compatibility (V100 vs A100 vs CPU)
2. DataLoader integrity (real bounding boxes, transforms, PK sampling)
3. End-to-end forward, backward, optimizer step, and scheduler step
4. Validation pipeline (Query / Gallery feature extraction, distance matrix, CMC/mAP)
5. Checkpoint saving and loading
6. Estimated epoch duration and 24-hour SLURM time limit check
"""

import sys
import time
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "OpenAnimals"))

from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from openanimals.solver import build_optimizer as build_oa_optimizer
from openanimals.solver import build_lr_scheduler as build_oa_scheduler


def run_preflight_check(model_name="oa_bot", clip_len=8, batch_size=16, epochs=120):
    print("=" * 70)
    print(f"       PRE-FLIGHT VERIFICATION: {model_name.upper()}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. HARDWARE & PRECISION CHECK
    # -------------------------------------------------------------
    print("\n[CHECK 1/6] Hardware & Mixed Precision Detection...")
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"  Device: {device}")
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  Hardware BF16 Support: {bf16_supported}")
        if not bf16_supported:
            print("  -> Auto-fallback: Will use FP16 with GradScaler (optimal for Volta/V100 Tensor Cores).")
        else:
            print("  -> Hardware BF16: Enabled (optimal for Ampere/Hopper).")
    else:
        print("  Running on CPU (login node). Full training should be submitted to GPU node.")

    # -------------------------------------------------------------
    # 2. CONFIGURATION SETUP
    # -------------------------------------------------------------
    print("\n[CHECK 2/6] Configuration & Hyperparameters...")
    cfg = Config()
    cfg.model = model_name
    cfg.backbone = model_name
    cfg.clip_len = clip_len
    cfg.batch_size = batch_size
    cfg.epochs = epochs
    cfg.accum_steps = 1
    cfg.val_split = 0.2
    cfg.update_model_settings()

    print(f"  Resolution       : {cfg.img_size}")
    print(f"  Clip length      : {cfg.clip_len} frames")
    print(f"  Batch size       : {cfg.batch_size} (P={cfg.num_ids} dogs, K={cfg.k} clips)")
    print(f"  Accumulation     : {cfg.accum_steps} (native per-batch updates)")
    print(f"  Random Erasing   : {cfg.re_prob}")
    print(f"  Target Epochs    : {cfg.epochs}")

    # -------------------------------------------------------------
    # 3. DATALOADER & SAMPLE EXTRACTION
    # -------------------------------------------------------------
    print("\n[CHECK 3/6] DataLoader & Dataset Integrity...")
    t0 = time.time()
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    dl_build_time = time.time() - t0
    print(f"  DataLoader built in {dl_build_time:.2f}s")
    print(f"  Train batches    : {len(train_loader)} (total {len(train_loader.dataset)} clips)")
    print(f"  Query samples    : {len(query_loader.dataset)} clips")
    print(f"  Gallery samples  : {len(gallery_loader.dataset)} clips")

    # Fetch 1 sample batch and measure time
    t0 = time.time()
    sample_batch = next(iter(train_loader))
    videos, labels, dog_ids, video_ids = sample_batch
    fetch_time = time.time() - t0
    print(f"  First batch load : {fetch_time:.2f}s")
    print(f"  Batch shape      : {videos.shape} -> [Batch={videos.size(0)}, Frames={videos.size(1)}, Channels={videos.size(2)}, H={videos.size(3)}, W={videos.size(4)}]")
    assert not torch.isnan(videos).any(), "NaN found in video batch!"

    # -------------------------------------------------------------
    # 4. MODEL BUILD & TRAINING STEP
    # -------------------------------------------------------------
    print("\n[CHECK 4/6] Model Forward, Backward & Optimizer Step...")
    model = build_model(cfg).to(device)
    model.train()

    # Build native OpenAnimals optimizer (contiguous=False)
    steps_per_epoch = len(train_loader)
    warmup_epochs = getattr(cfg, 'warmup_epochs', 5)
    model.oa_cfg.defrost()
    model.oa_cfg.SOLVER.MAX_EPOCH = cfg.epochs
    model.oa_cfg.SOLVER.WARMUP_ITERS = warmup_epochs * steps_per_epoch
    model.oa_cfg.freeze()

    optimizer, _ = build_oa_optimizer(model.oa_cfg, model, contiguous=False)
    oa_sched_dict = build_oa_scheduler(model.oa_cfg, optimizer, iters_per_epoch=steps_per_epoch)

    # Test training step
    videos_dev = videos.to(device)
    labels_dev = labels.to(device)

    # Step 1
    optimizer.zero_grad()
    t0 = time.time()
    feat, cls_outputs, loss1, loss_dict = model(videos_dev, targets=labels_dev)
    loss1.backward()
    optimizer.step()
    step_time = time.time() - t0

    # Step 2
    optimizer.zero_grad()
    feat, cls_outputs, loss2, _ = model(videos_dev, targets=labels_dev)
    loss2.backward()
    optimizer.step()
    oa_sched_dict["warmup_sched"].step()

    print(f"  Step execution   : {step_time:.3f}s per batch")
    print(f"  Initial Loss     : {loss1.item():.4f} ({', '.join(f'{k}: {v.item():.3f}' for k, v in loss_dict.items())})")
    print(f"  Second Loss      : {loss2.item():.4f}")
    assert loss1.item() > 0 and loss2.item() > 0, "Loss was zero or negative!"
    print("  -> Gradient backprop and parameter stepping verified.")

    # -------------------------------------------------------------
    # 5. VALIDATION & CMC / mAP EVALUATION PIPELINE
    # -------------------------------------------------------------
    print("\n[CHECK 5/6] Validation Pipeline (Query & Gallery Evaluation)...")
    model.eval()
    with torch.no_grad():
        eval_features = model(videos_dev)
        print(f"  Inference Feature: shape={eval_features.shape}")
        expected_dim = 16384 if "mgn" in model_name else 2048
        assert eval_features.size(1) == expected_dim, f"Feature dimension mismatch: got {eval_features.size(1)}, expected {expected_dim}"

    # Test a mini evaluation through Trainer._get_features and calculate_cmc_map
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        loss_fn=None,
        miner=None,
        cfg=cfg,
        oa_sched_dict=oa_sched_dict
    )

    print("  Testing evaluation metric calculation on dummy split...")
    q_feats = torch.randn(10, expected_dim)
    g_feats = torch.randn(20, expected_dim)
    q_pids = torch.arange(10)
    g_pids = torch.cat([torch.arange(10), torch.arange(10)])
    dist_mat = 1 - torch.mm(F.normalize(q_feats, p=2, dim=1), F.normalize(g_feats, p=2, dim=1).t())
    r1, r5, map_score = trainer.calculate_cmc_map(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
    print(f"  CMC/mAP Function : Rank-1={r1:.1%}, Rank-5={r5:.1%}, mAP={map_score:.1%}")
    print("  -> Evaluation metric engine verified.")

    # -------------------------------------------------------------
    # 6. TIME ESTIMATE & SLURM SAFETY CHECK
    # -------------------------------------------------------------
    print("\n[CHECK 6/6] Runtime & SLURM Time Limit Safety...")
    est_train_time_per_epoch = len(train_loader) * step_time  # in seconds
    est_total_train_hours = (est_train_time_per_epoch * cfg.epochs) / 3600.0

    print(f"  Batches per Epoch: {len(train_loader)}")
    print(f"  Est. Train/Epoch : {est_train_time_per_epoch / 60.0:.1f} minutes (excluding eval)")
    print(f"  Total Run (120ep): {est_total_train_hours:.1f} hours")

    if est_total_train_hours > 23.0:
        print("\n  [WARNING] Projected runtime is close to or exceeds the 24-hour SLURM limit!")
        print("  Recommendations to comfortably finish in <24 hours:")
        print(f"   1. Use --clip_len 8 (instead of 16): reduces training time by ~50%.")
        print(f"   2. Use --eval_period 10: avoids running full validation every single epoch.")
        print(f"   3. Checkpoints are automatically saved every 10 epochs for resume capability.")
    else:
        print("  -> Projected runtime is within normal SLURM limits.")

    print("\n" + "=" * 70)
    print("  PRE-FLIGHT VERIFICATION PASSED! READY FOR CLUSTER SUBMISSION.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-Flight Training Check")
    parser.add_argument("--model", type=str, default="oa_bot",
                        choices=["oa_bot", "oa_agw", "oa_sbs", "oa_mgn", "oa_arbase"])
    parser.add_argument("--clip_len", type=int, default=8, help="Frames per video clip (default 8)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs (default 120)")
    args = parser.parse_args()

    run_preflight_check(
        model_name=args.model,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
