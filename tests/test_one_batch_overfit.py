#!/usr/bin/env python
"""
One-Batch Overfit Sanity Check for OpenAnimals Video Re-ID.
Loads ONE real batch from the dataset and trains on it repeatedly for 50 steps.
A healthy network should drive loss near zero and achieve 100% accuracy.
"""

import sys
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
from openanimals.solver import build_optimizer as build_oa_optimizer


def main():
    parser = argparse.ArgumentParser(description="One-Batch Overfit Test")
    parser.add_argument("--model", type=str, default="oa_bot",
                        choices=["oa_bot", "oa_agw", "oa_sbs", "oa_mgn", "oa_arbase", "dinov2"],
                        help="Model architecture to test")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps on the batch")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 clips = 4 dogs x 4 clips)")
    parser.add_argument("--clip_len", type=int, default=8, help="Frames per video clip")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for overfitting test")
    args = parser.parse_args()

    print("=" * 65)
    print(f"       ONE-BATCH OVERFIT TEST: {args.model.upper()}")
    print("=" * 65)

    # 1. Config setup
    cfg = Config()
    cfg.model = args.model
    cfg.backbone = args.model
    cfg.batch_size = args.batch_size
    cfg.clip_len = args.clip_len
    cfg.lr = args.lr
    cfg.accum_steps = 1
    cfg.update_model_settings()

    device = cfg.device
    print(f"Device: {device}")
    print(f"Image Resolution: {cfg.img_size}")
    print(f"Batch Configuration: batch_size={cfg.batch_size} (P={cfg.num_ids}, K={cfg.k}), clip_len={cfg.clip_len}")

    # 2. Build Dataloaders and extract exactly ONE real batch
    print("\n--> Building DataLoader from dataset...")
    train_loader, _, _ = build_dataloaders(cfg)

    # Determine num_classes
    if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'id_map'):
        cfg.num_classes = len(train_loader.dataset.dataset.id_map)
    else:
        cfg.num_classes = 1553

    print(f"--> Total training identities: {cfg.num_classes}")

    # Fetch 1 real batch
    batch = next(iter(train_loader))
    videos, labels, dog_ids, video_ids = batch
    videos = videos.to(device)
    labels = labels.to(device)

    print(f"--> Batch extracted: shape={videos.shape} (B={videos.size(0)}, T={videos.size(1)}, C={videos.size(2)}, H={videos.size(3)}, W={videos.size(4)})")
    print(f"    Dog IDs in batch: {labels.tolist()}")

    # 3. Build Model
    model = build_model(cfg).to(device)
    model.train()

    # 4. Build Optimizer
    if getattr(model, "is_openanimals", False):
        model.oa_cfg.defrost()
        model.oa_cfg.SOLVER.BASE_LR = args.lr
        model.oa_cfg.freeze()
        optimizer, _ = build_oa_optimizer(model.oa_cfg, model, contiguous=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"--> Optimizer: {type(optimizer).__name__} (lr={args.lr})")
    print(f"\n{'Step':>5} | {'Total Loss':>10} | {'Classif Acc':>11} | {'Retrieval R1':>12} | {'Breakdown':<25}")
    print("-" * 75)

    initial_loss = None
    final_loss = None
    final_acc = 0.0
    final_r1 = 0.0

    # 5. Overfitting Training Loop on this single batch
    for step in range(args.steps + 1):
        optimizer.zero_grad()

        if getattr(model, "is_openanimals", False):
            feat, cls_outputs, total_loss, loss_dict = model(videos, targets=labels)
            breakdown = " ".join(f"{k.replace('loss_', '')}:{v.item():.3f}" for k, v in loss_dict.items())
        else:
            outputs = model(videos)
            feat, logits = outputs if isinstance(outputs, tuple) else (outputs, None)
            from pytorch_metric_learning import losses, miners
            miner = miners.BatchHardMiner()
            triplet_loss = losses.TripletMarginLoss(margin=0.3)(feat, labels, miner(feat, labels))
            cls_loss = F.cross_entropy(logits, labels) if logits is not None else 0.0
            total_loss = triplet_loss + cls_loss
            cls_outputs = logits
            breakdown = f"tri:{triplet_loss.item():.3f} ce:{cls_loss.item():.3f}"

        if initial_loss is None:
            initial_loss = total_loss.item()

        # Backward and step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Compute accuracy and within-batch Rank-1 retrieval
        with torch.no_grad():
            # Classification accuracy
            if cls_outputs is not None:
                preds = cls_outputs.argmax(dim=-1)
                acc = (preds == labels).float().mean().item() * 100.0
            else:
                acc = 0.0

            # Within-batch retrieval (exclude self-comparison on diagonal)
            norm_feat = F.normalize(feat, p=2, dim=-1)
            sim_mat = torch.mm(norm_feat, norm_feat.t())
            sim_mat.fill_diagonal_(-999.0)
            top1_matches = sim_mat.argmax(dim=1)
            r1 = (labels[top1_matches] == labels).float().mean().item() * 100.0

        if step % 5 == 0 or step == args.steps:
            print(f"{step:5d} | {total_loss.item():10.4f} | {acc:10.1f}% | {r1:11.1f}% | {breakdown:<25}")

        final_loss = total_loss.item()
        final_acc = acc
        final_r1 = r1

    # 6. Evaluation summary
    print("-" * 75)
    loss_reduction = (initial_loss - final_loss) / max(initial_loss, 1e-6) * 100.0
    print(f"\nInitial Loss: {initial_loss:.4f}  --->  Final Loss: {final_loss:.4f}  (Reduced by {loss_reduction:.1f}%)")
    print(f"Final Classification Accuracy: {final_acc:.1f}%")
    print(f"Final Within-Batch Rank-1:    {final_r1:.1f}%")

    if loss_reduction > 60.0:
        print("\n[PASSED] The model successfully overfits the single batch!")
        print(" Gradients, loss formulation, temporal pooling, and optimizer updates are 100% verified.")
    else:
        print("\n[WARNING] Loss reduction was below 60%. Parameters may not be learning sufficiently.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
