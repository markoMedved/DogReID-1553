import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR and ("classifier" in key):
            lr = cfg.SOLVER.BASE_LR * 2

        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})

    if cfg.SOLVER.OPTIMIZER_NAME == "SGD":
        return torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    if cfg.SOLVER.OPTIMIZER_NAME == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    return getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
