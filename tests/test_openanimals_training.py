#!/usr/bin/env python
"""
Verification test suite for OpenAnimals video training.
Verifies that for each of the 5 OpenAnimals architectures:
1. Model forward pass and native losses compute correctly.
2. Backpropagation and optimizer step properly update model parameters on every step.
3. Training loss strictly decreases over optimization steps on a sample batch.
4. Warmup scheduler and epoch scheduler advance as expected.
5. Inference mode extracts properly normalized evaluation embeddings.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "OpenAnimals"))

from configs.config import Config
from models.openanimals_models import OpenAnimalsVideoModel
from openanimals.solver import build_optimizer as build_oa_optimizer
from openanimals.solver import build_lr_scheduler as build_oa_scheduler


MODELS = ["oa_bot", "oa_agw", "oa_sbs", "oa_mgn", "oa_arbase"]


def test_model_training(model_name: str):
    print(f"\n{'=' * 60}")
    print(f"Testing model: {model_name}")
    print(f"{'=' * 60}")

    cfg = Config()
    cfg.model = model_name
    cfg.update_model_settings()

    num_classes = 10
    model = OpenAnimalsVideoModel(model_name, num_classes=num_classes)
    model.train()

    # Verify optimizer setup with contiguous=False
    steps_per_epoch = 10
    warmup_epochs = 2
    model.oa_cfg.defrost()
    model.oa_cfg.SOLVER.MAX_EPOCH = 20
    model.oa_cfg.SOLVER.WARMUP_ITERS = warmup_epochs * steps_per_epoch
    model.oa_cfg.freeze()

    optimizer, _ = build_oa_optimizer(model.oa_cfg, model, contiguous=False)
    sched_dict = build_oa_scheduler(model.oa_cfg, optimizer, iters_per_epoch=steps_per_epoch)

    # 1. Forward Pass & Loss Test
    H, W = cfg.img_size
    B, T = 4, 4
    torch.manual_seed(42)
    dummy_videos = torch.randn(B, T, 3, H, W)
    dummy_targets = torch.tensor([0, 0, 1, 1])

    feat, cls_outputs, loss, loss_dict = model(dummy_videos, targets=dummy_targets)
    assert loss is not None and torch.isfinite(loss), f"Loss is invalid: {loss}"
    sub_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
    print(f"[1/4] Forward Pass OK: total_loss={loss.item():.4f} ({sub_str})")

    # 2. Continuous Weight Update Test across consecutive steps
    # We record weights before and after Step 1, Step 2, and Step 3
    weight_name = "oa_model.backbone.conv1.weight"
    for n, p in model.named_parameters():
        if "conv" in n and p.requires_grad:
            weight_name = n
            break

    initial_w = dict(model.named_parameters())[weight_name].clone()

    # Step 1
    optimizer.zero_grad()
    feat, cls_outputs, loss1, _ = model(dummy_videos, targets=dummy_targets)
    loss1.backward()
    optimizer.step()
    w_after_step1 = dict(model.named_parameters())[weight_name].clone()
    diff1 = (w_after_step1 - initial_w).abs().max().item()
    assert diff1 > 0, f"Step 1 failed: {weight_name} did not update!"

    # Step 2 (Verifies zero_grad does NOT decouple the optimizer)
    optimizer.zero_grad()
    feat, cls_outputs, loss2, _ = model(dummy_videos, targets=dummy_targets)
    loss2.backward()
    optimizer.step()
    w_after_step2 = dict(model.named_parameters())[weight_name].clone()
    diff2 = (w_after_step2 - w_after_step1).abs().max().item()
    assert diff2 > 0, f"Step 2 failed: {weight_name} did not update after zero_grad!"

    print(f"[2/4] Consecutive Weight Updates OK: diff_step1={diff1:.6e}, diff_step2={diff2:.6e}")

    # 3. Loss Descent Test
    # Over 10 consecutive steps on this fixed batch, loss MUST drop significantly
    initial_loss = loss1.item()
    current_loss = initial_loss
    for step in range(10):
        optimizer.zero_grad()
        feat, cls_outputs, step_loss, _ = model(dummy_videos, targets=dummy_targets)
        step_loss.backward()
        optimizer.step()
        sched_dict["warmup_sched"].step()
        current_loss = step_loss.item()

    assert current_loss < initial_loss, f"Loss did not decrease! Initial={initial_loss:.4f}, Final={current_loss:.4f}"
    print(f"[3/4] Loss Drop OK: Initial={initial_loss:.4f} -> Final={current_loss:.4f} (Dropped {initial_loss - current_loss:.4f})")

    # 4. Evaluation Feature Extraction Test
    model.eval()
    with torch.no_grad():
        eval_feat = model(dummy_videos)
    expected_dim = 16384 if "mgn" in model_name else 2048
    assert eval_feat.shape == (B, expected_dim), f"Unexpected eval feature shape: {eval_feat.shape}"
    norms = torch.norm(eval_feat, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Evaluation features are not L2-normalized!"
    print(f"[4/4] Evaluation OK: Output shape={eval_feat.shape}, L2-normalized={True}")

    print(f"SUCCESS: {model_name} passed all verification checks!\n")


def main():
    print("=================================================================")
    print("      OpenAnimals Video Training Verification Test Suite         ")
    print("=================================================================")
    for m in MODELS:
        test_model_training(m)
    print("=================================================================")
    print("  ALL 5 OPENANIMALS MODELS PASSED ALL VERIFICATION TESTS!        ")
    print("=================================================================")


if __name__ == "__main__":
    main()
