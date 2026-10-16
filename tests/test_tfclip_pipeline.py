import torch
from configs.config import Config
from models.model_factory import build_model
from models.freezing import apply_freezing, trainable_report


def test_tfclip_forward_backward():
    print("=== Testing TF-CLIP Pipeline Integration ===")
    cfg = Config()
    cfg.model = "tfclip"
    cfg.backbone = "tfclip"
    cfg.img_size = (224, 224)
    cfg.clip_len = 8
    cfg.num_classes = 10
    cfg.lr = 5e-5
    cfg.backbone_lr_factor = 0.1
    cfg.update_model_settings()

    print("[1] Building TF-CLIP model...")
    model = build_model(cfg)
    assert getattr(model, "is_tfclip", False), "Model is not marked as is_tfclip!"
    print("    Model instantiated successfully.")

    print("[2] Checking Freezing and Trainable parameters...")
    model = apply_freezing(model, cfg)
    report = trainable_report(model)
    print(report)

    # Initialize dummy memory for testing
    model.clip_memory.copy_(torch.randn(10, 512))
    model.memory_initialized.copy_(torch.tensor(True, dtype=torch.bool))

    print("[3] Testing Training Forward Pass...")
    model.train()
    B, T, C, H, W = 4, 8, 3, 224, 224
    dummy_videos = torch.randn(B, T, C, H, W)
    dummy_targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    outputs = model(dummy_videos, targets=dummy_targets)
    assert "scores" in outputs and "feats" in outputs and "logits" in outputs, "Invalid output structure!"
    print("    Training output keys:", list(outputs.keys()))
    print("    Logits shape:", outputs["logits"].shape)
    assert outputs["logits"].shape == (B, 10), f"Expected logits shape {(B, 10)}, got {outputs['logits'].shape}"

    print("[4] Testing Loss Computation...")
    total_loss, loss_triplet, loss_id = model.compute_loss(outputs, dummy_targets)
    print(f"    Total Loss: {total_loss.item():.4f} | Triplet: {loss_triplet.item():.4f} | ID: {loss_id.item():.4f}")
    assert total_loss.item() > 0, "Loss must be positive!"

    print("[5] Testing Backward Pass...")
    total_loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients computed!"
    print(f"    Gradients computed on {len(grad_norms)} tensors.")

    print("[6] Testing Inference Forward Pass...")
    model.eval()
    with torch.no_grad():
        features = model(dummy_videos)
    print(f"    Inference feature shape: {features.shape}")
    assert features.shape == (B, 2048), f"Expected feature shape {(B, 2048)}, got {features.shape}"
    norms = torch.norm(features, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Features must be L2 normalized!"
    print("    L2 normalization verified.")

    print("[7] Testing Optimizer Parameter Groups...")
    param_groups = model.get_parameter_groups(base_lr=cfg.lr, backbone_lr_factor=cfg.backbone_lr_factor)
    assert len(param_groups) == 2, f"Expected 2 param groups (backbone, head), got {len(param_groups)}"
    bb_lr = param_groups[0]["lr"]
    hd_lr = param_groups[1]["lr"]
    print(f"    Backbone LR: {bb_lr:.2e} | Head LR: {hd_lr:.2e}")
    assert abs(bb_lr - 5e-6) < 1e-8, f"Expected backbone LR 5e-6, got {bb_lr}"
    assert abs(hd_lr - 5e-5) < 1e-8, f"Expected head LR 5e-5, got {hd_lr}"

    print("\n>>> ALL TF-CLIP PIPELINE TESTS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_tfclip_forward_backward()
