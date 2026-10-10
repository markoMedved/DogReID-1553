"""Learning rate schedule with linear warmup.

Follows Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person
Re-Identification" (CVPRW 2019 / IEEE TMM 2020), Sec. 3.2: the learning rate is
ramped linearly over the first epochs before step decay.
"""

from bisect import bisect_right
import math

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Multi-step decay preceded by a linear or constant warmup phase."""

    def __init__(self, optimizer, milestones=(40, 70), gamma=0.1,
                 warmup_factor=0.01, warmup_iters=10, warmup_method="linear",
                 backbone_drop_epoch=None, backbone_drop_factor=0.1,
                 last_epoch=-1):

        if list(milestones) != sorted(milestones):
            raise ValueError(f"Milestones must be increasing, got {milestones}")
        if warmup_method not in ("constant", "linear"):
            raise ValueError(f"Unknown warmup_method: {warmup_method!r}")

        self.milestones = list(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.backbone_drop_epoch = backbone_drop_epoch
        self.backbone_drop_factor = backbone_drop_factor

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # --- Warmup Phase ---
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        # --- Step Decay ---
        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            lr = base_lr * warmup_factor * (self.gamma ** bisect_right(self.milestones, self.last_epoch))
            is_backbone = self.optimizer.param_groups[i].get('is_backbone', (i == 0 and len(self.base_lrs) > 1))
            if is_backbone and self.backbone_drop_epoch is not None and self.last_epoch >= self.backbone_drop_epoch:
                lr = lr * self.backbone_drop_factor
            lrs.append(lr)
        return lrs


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing decay preceded by linear warmup and optional delay phase.

    Matches OpenAnimals / FastReID CosineAnnealingLR with warmup and delay_epochs:
    - Linear warmup from warmup_factor * base_lr to base_lr for warmup_epochs.
    - Constant base_lr until delay_epochs.
    - Cosine annealing from delay_epochs to max_epochs decaying to eta_min.
    """

    def __init__(
        self,
        optimizer,
        max_epochs: int = 120,
        delay_epochs: int = 60,
        eta_min: float = 7e-7,
        warmup_factor: float = 0.1,
        warmup_epochs: int = 10,
        backbone_drop_epoch: int = None,
        backbone_drop_factor: float = 0.1,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.delay_epochs = delay_epochs
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.backbone_drop_epoch = backbone_drop_epoch
        self.backbone_drop_factor = backbone_drop_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            warmup_factor = self.warmup_factor * (1.0 - alpha) + alpha
            raw_lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]
        elif self.last_epoch < self.delay_epochs:
            raw_lrs = [base_lr for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.delay_epochs) / max(1, self.max_epochs - self.delay_epochs)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            raw_lrs = [
                self.eta_min + (base_lr - self.eta_min) * cosine_factor
                for base_lr in self.base_lrs
            ]

        for i in range(len(raw_lrs)):
            is_backbone = self.optimizer.param_groups[i].get('is_backbone', (i == 0 and len(self.base_lrs) > 1))
            if is_backbone and self.backbone_drop_epoch is not None and self.last_epoch >= self.backbone_drop_epoch:
                raw_lrs[i] = raw_lrs[i] * self.backbone_drop_factor

        return raw_lrs


def build_scheduler(optimizer, cfg):
    """Instantiate the schedule from the training configuration."""
    sched_type = getattr(cfg, "lr_sched", None)
    model_name = str(getattr(cfg, "backbone", getattr(cfg, "model", ""))).lower()
    drop_epoch = getattr(cfg, "backbone_drop_epoch", None)
    drop_factor = getattr(cfg, "backbone_drop_factor", 0.1)

    if sched_type == "cosine" or (sched_type is None and model_name in ("arbase", "sbs", "mgn")):
        delay = getattr(cfg, "lr_delay_epochs", 30 if "sbs" in model_name else 60)
        return WarmupCosineAnnealingLR(
            optimizer,
            max_epochs=getattr(cfg, "epochs", 120),
            delay_epochs=delay,
            eta_min=getattr(cfg, "eta_min", 7e-7),
            warmup_factor=getattr(cfg, "warmup_factor", 0.1),
            warmup_epochs=getattr(cfg, "warmup_epochs", 10),
            backbone_drop_epoch=drop_epoch,
            backbone_drop_factor=drop_factor,
        )

    return WarmupMultiStepLR(
        optimizer,
        milestones=getattr(cfg, "lr_milestones", (40, 70)),
        gamma=getattr(cfg, "lr_gamma", 0.1),
        warmup_factor=getattr(cfg, "warmup_factor", 0.01),
        warmup_iters=getattr(cfg, "warmup_epochs", 10),
        backbone_drop_epoch=drop_epoch,
        backbone_drop_factor=drop_factor,
    )
