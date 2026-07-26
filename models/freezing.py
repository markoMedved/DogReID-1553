import torch.nn as nn


# --- Parameter Unfreezing Helpers ---

def _unfreeze(module):
    """Enable gradients on an nn.Module or a bare nn.Parameter."""
    if module is None:
        return
    if isinstance(module, nn.Module):
        for p in module.parameters():
            p.requires_grad = True
    elif hasattr(module, "requires_grad_"):
        module.requires_grad_(True)


def _core_network(model):
    """Return the backbone network, unwrapping the adapter if present."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    for attr in ("net", "visual"):
        inner = getattr(backbone, attr, None)
        if isinstance(inner, nn.Module):
            return inner

    return backbone


def _last_n(modules, n):
    """Return the final n modules, or none at all when n <= 0.

    Slicing with [-0:] returns the whole sequence, so n <= 0 is handled
    explicitly to mean "unfreeze no backbone blocks".
    """
    return modules[-n:] if n > 0 else []


def _unfreeze_last_blocks(net, n=2):
    """Unfreeze the final n blocks and the final normalization layer.

    Swin and ConvNeXt are unfrozen one stage at a time rather than one block at
    a time, so n is not used for those backbones.

    Returns False if the backbone layout is not recognized.
    """
    if net is None:
        return False

    # --- DINOv2 / timm ViT ---
    if hasattr(net, "blocks"):
        for block in _last_n(net.blocks, n):
            _unfreeze(block)
        if n > 0:
            _unfreeze(getattr(net, "norm", None))
        return True

    # --- Torchvision ViT ---
    encoder = getattr(net, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers"):
        for layer in _last_n(encoder.layers, n):
            _unfreeze(layer)
        if n > 0:
            _unfreeze(getattr(encoder, "ln", None))
        return True

    # --- Swin Transformer (hierarchical stages) ---
    if hasattr(net, "layers"):
        if n > 0:
            _unfreeze(net.layers[-1])
            _unfreeze(getattr(net, "norm", None))
        return True

    # --- OSNet (Torchreid) ---
    # Convolutional stages conv1..conv5 plus an fc bottleneck; unfrozen one
    # stage at a time, so n is not used
    if hasattr(net, "conv5") and hasattr(net, "global_avgpool"):
        if n > 0:
            _unfreeze(net.conv5)
            _unfreeze(getattr(net, "fc", None))
        return True

    # --- ConvNeXt ---
    if hasattr(net, "stages"):
        if n > 0:
            _unfreeze(net.stages[-1])
            _unfreeze(getattr(net, "norm_pre", None))
            _unfreeze(getattr(net, "head", None))
        return True

    return False


def apply_freezing(model, cfg):
    """Set requires_grad across the model according to cfg.full_finetune."""

    # --- Full Fine-Tuning ---
    if getattr(cfg, "full_finetune", False):
        for p in model.parameters():
            p.requires_grad = True
        return model

    # --- Partial Fine-Tuning ---
    # Freeze the whole model, then selectively unfreeze the deepest layers
    for p in model.parameters():
        p.requires_grad = False

    net = _core_network(model)
    if not _unfreeze_last_blocks(net, n=getattr(cfg, "unfreeze_blocks", 2)):
        raise RuntimeError(
            f"Unrecognized backbone layout: {type(net).__name__}. Training would "
            f"proceed with a fully frozen backbone. Add the layout to "
            f"_unfreeze_last_blocks() or set cfg.full_finetune = True."
        )

    # --- Always-Trainable Components ---
    # Temporal pooling, BN necks and identity heads, for both the current
    # VideoReID layout (ModuleLists) and the single-branch builders.
    for attr in ("pools", "heads", "temporal_pool", "temporal_attn",
                 "bn", "bottleneck", "classifier"):
        _unfreeze(getattr(model, attr, None))

    # The JPM local branch holds its own copy of the final block
    jpm = getattr(model, "jpm", None)
    if jpm is not None:
        for attr in ("global_block", "global_norm", "local_block", "local_norm"):
            _unfreeze(getattr(jpm, attr, None))

    return model


def trainable_report(model):
    """Summarize how many parameters the current freezing policy leaves trainable."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return f"[freeze] trainable {trainable:,} / {total:,} params ({pct:.1f}%)"
