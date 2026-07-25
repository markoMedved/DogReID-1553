"""Learning rate schedule with linear warmup.

Follows Luo et al., "Bag of Tricks and a Strong Baseline for Deep Person
Re-Identification" (CVPRW 2019 / IEEE TMM 2020), Sec. 3.2: the learning rate is
ramped linearly over the first epochs before step decay.
"""

from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Multi-step decay preceded by a linear or constant warmup phase."""

    def __init__(self, optimizer, milestones=(40, 70), gamma=0.1,
                 warmup_factor=0.01, warmup_iters=10, warmup_method="linear",
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
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, cfg):
    """Instantiate the schedule from the training configuration."""
    return WarmupMultiStepLR(
        optimizer,
        milestones=getattr(cfg, "lr_milestones", (40, 70)),
        gamma=getattr(cfg, "lr_gamma", 0.1),
        warmup_factor=getattr(cfg, "warmup_factor", 0.01),
        warmup_iters=getattr(cfg, "warmup_epochs", 10),
    )
