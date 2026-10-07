import argparse
import torch
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners

from models.freezing import apply_freezing, trainable_report
from solver.warmup import build_scheduler


def main():
    # ------------------------------------------------
    # ARGUMENT PARSING
    # ------------------------------------------------
    parser = argparse.ArgumentParser(description="Dog Re-ID Training")

    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--margin', type=float, default=None, help='Triplet loss margin')
    parser.add_argument('--weight_decay', type=float, default=None, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (P*K)')
    parser.add_argument('--k', type=int, default=None, help='Clips per dog ID')
    parser.add_argument('--model', type=str, default=None, help="Backbone: 'dinov2', 'swin', 'vit', 'convnetxt'")
    parser.add_argument('--world', type=str, default=None, help="'closed' or 'open'")
    parser.add_argument('--clip_len', type=int, default=None, help='Frames per video clip')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--val_split', type=float, default=None, help='Validation split ratio (e.g., 0.2 for 20%% validation)')
    parser.add_argument('--eval_period', type=int, default=None, help='Evaluation frequency in epochs (default from config: 1)')

    # Re-ID methodology flags
    parser.add_argument(
        '--reid_method', 
        type=str, 
        default=None,
        choices=['bot', 'transreid', 'baseline'],
        help="Re-ID method: 'bot', 'transreid'; 'baseline' uses legacy baseline builders"
    )
    parser.add_argument(
        '--pooling_type', 
        type=str, 
        default=None, 
        choices=['attention', 'mean', 'max'],
        help="Temporal aggregation method: 'attention', 'mean', or 'max'"
    )
    parser.add_argument(
        '--full_finetune', 
        dest='full_finetune', 
        action='store_true', 
        default=None, 
        help='Unfreeze the entire backbone for end-to-end fine-tuning'
    )
    parser.add_argument(
        '--no_full_finetune', 
        dest='full_finetune', 
        action='store_false', 
        help='Train only the last unfreeze_blocks backbone blocks'
    )
    parser.add_argument(
        '--unfreeze_blocks', 
        type=int, 
        default=None,
        help='Trailing backbone blocks to train when not full fine-tuning'
    )
    parser.add_argument(
        '--use_id_loss', 
        dest='use_id_loss', 
        action='store_true', 
        default=None, 
        help='Enable Identity Classification Loss alongside Triplet Loss'
    )
    parser.add_argument(
        '--no_use_id_loss', 
        dest='use_id_loss', 
        action='store_false', 
        help='Disable Identity Classification Loss'
    )
    parser.add_argument(
        '--mask_dog', 
        action='store_true', 
        default=False, 
        help='Mask out the dog for the background-only diagnostic experiment'
    )
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default=None, 
        help='Optimizer to use: adam, adamw, or sgd'
    )
    parser.add_argument(
        '--accum_steps', 
        type=int, 
        default=None, 
        help='Number of gradient accumulation steps (default 1 for OpenAnimals)'
    )
    parser.add_argument(
        '--backbone_lr_factor',
        type=float,
        default=None,
        help='Multiplier for backbone LR relative to base LR (default: 1.0 to match OpenAnimals)'
    )
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help='Resume training from the latest checkpoint in output_dir'
    )

    args = parser.parse_args()

    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()

    # --- Command-Line Overrides ---
    if args.resume:
        cfg.resume = True
    if args.mask_dog:
        cfg.mask_dog = True
    if args.model is not None:
        cfg.model = args.model
        cfg.backbone = args.model
    if args.world is not None: cfg.world = args.world
    if args.clip_len is not None: cfg.clip_len = args.clip_len
    if args.lr is not None: cfg.lr = args.lr
    if args.margin is not None: cfg.margin = args.margin
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.batch_size is not None: cfg.batch_size = args.batch_size
    if args.k is not None: cfg.k = args.k
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.val_split is not None: cfg.val_split = args.val_split
    if args.eval_period is not None: cfg.eval_period = args.eval_period
    if args.accum_steps is not None: cfg.accum_steps = args.accum_steps
    if args.reid_method is not None:
        cfg.reid_method = None if args.reid_method == 'baseline' else args.reid_method
    if args.pooling_type is not None: cfg.pooling_type = args.pooling_type
    if args.full_finetune is not None: cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None: cfg.unfreeze_blocks = args.unfreeze_blocks
    if args.use_id_loss is not None: cfg.use_id_loss = args.use_id_loss

    if args.optimizer is not None: cfg.optimizer = args.optimizer
    if args.backbone_lr_factor is not None: cfg.backbone_lr_factor = args.backbone_lr_factor

    # Re-ID BoT hyperparameter adjustments if reid_method is active and not overridden
    if cfg.reid_method in ('bot', 'transreid'):
        if args.epochs is None: cfg.epochs = getattr(cfg, "epochs", 120)
        if args.lr is None: cfg.lr = getattr(cfg, "lr", 3.5e-04)
        if args.clip_len is None: cfg.clip_len = 8
        if args.accum_steps is None: cfg.accum_steps = getattr(cfg, "accum_steps", 1)

        if not any(m < cfg.epochs for m in cfg.lr_milestones):
            raise ValueError(
                f"lr_milestones {cfg.lr_milestones} all fall outside epochs={cfg.epochs}; "
                f"the learning rate would never decay. Rescale them together."
            )

    cfg.update_model_settings()
    if args.optimizer is not None: cfg.optimizer = args.optimizer
    if args.backbone_lr_factor is not None: cfg.backbone_lr_factor = args.backbone_lr_factor
    if args.lr is not None: cfg.lr = args.lr
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.val_split is not None: cfg.val_split = args.val_split
    if args.eval_period is not None: cfg.eval_period = args.eval_period
    if args.full_finetune is not None: cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None: cfg.unfreeze_blocks = args.unfreeze_blocks
    cfg.refresh_run_name(make_dir=True)
    cfg.display()

    # ------------------------------------------------
    # BUILD DATA LOADERS
    # ------------------------------------------------
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

    # Dynamic num_classes calculation
    if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'id_map'):
        cfg.num_classes = len(train_loader.dataset.dataset.id_map)
    elif hasattr(train_loader.dataset, 'dog_ids'):
        cfg.num_classes = len(set(train_loader.dataset.dog_ids))
    elif hasattr(train_loader.dataset, 'labels'):
        cfg.num_classes = len(set(train_loader.dataset.labels))
    elif hasattr(train_loader.dataset, 'pids'):
        cfg.num_classes = len(set(train_loader.dataset.pids))
    elif hasattr(train_loader.dataset, 'classes'):
        cfg.num_classes = len(train_loader.dataset.classes)
    else:
        cfg.num_classes = len(train_loader.dataset.targets if hasattr(train_loader.dataset, 'targets') else train_loader.dataset)

    print(f"--> Total training dog identities (num_classes): {cfg.num_classes}")

    # If id_loss is disabled for baseline, reset num_classes = 0 so the classifier head isn't built
    if not cfg.use_id_loss and cfg.reid_method not in ('bot', 'transreid'):
        cfg.num_classes = 0

    # ------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------
    model = build_model(cfg).to(cfg.device)

    # Apply freezing
    model = apply_freezing(model, cfg)
    print(trainable_report(model))

    # ------------------------------------------------
    # OPTIMIZER & SCHEDULER
    # ------------------------------------------------
    oa_sched_dict = None

    def is_pretrained(name):
        if 'NL_' in name or 'nonlocal' in name.lower():
            return False
        return (
            name.startswith('backbone')
            or name.startswith('jpm.')
            or name.startswith('oa_model.backbone')
            or name.startswith('oa_model.b1')
            or name.startswith('oa_model.b2')
            or name.startswith('oa_model.b3')
            or name.startswith('shared_base')
            or name.startswith('b1.')
            or name.startswith('b2.')
            or name.startswith('b3.')
        )

    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and is_pretrained(n)
    ]

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not is_pretrained(n)
    ]

    bb_lr_factor = float(getattr(cfg, "backbone_lr_factor", 1.0))
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr * bb_lr_factor})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr})

    if not param_groups:
        raise ValueError(
            "No trainable parameters found for optimizer! "
            "Ensure at least part of the backbone or head has requires_grad=True."
        )

    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[optim] Backbone: {n_bb:,} params, base LR = {cfg.lr * bb_lr_factor:.2e} ({bb_lr_factor}x) | Head: {n_hd:,} params, base LR = {cfg.lr:.2e}")
    opt_name = str(getattr(cfg, "optimizer", "adam")).lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=getattr(cfg, "weight_decay", 5e-4))
        print(f"[optim] Using SGD optimizer (momentum=0.9, weight_decay={getattr(cfg, 'weight_decay', 5e-4)})")
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
        print(f"[optim] Using AdamW optimizer (weight_decay={cfg.weight_decay})")
    else:
        optimizer = torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
        print(f"[optim] Using Adam optimizer (weight_decay={cfg.weight_decay})")

    # --- BUILD SCHEDULER ---
    scheduler = None
    oa_sched_dict = None
    if getattr(model, "is_openanimals", False):
        try:
            from openanimals.solver.build import build_lr_scheduler
            oa_sched_dict = build_lr_scheduler(model.oa_cfg, optimizer, len(train_loader))
            print(f"[scheduler] Initialized OpenAnimals native scheduler: {model.oa_cfg.SOLVER.SCHED}")
        except Exception as e:
            print(f"[scheduler] Warning: failed to build OpenAnimals scheduler ({e}), falling back to WarmupMultiStepLR")
            scheduler = build_scheduler(optimizer, cfg)
    elif cfg.reid_method in ('bot', 'transreid'):
        scheduler = build_scheduler(optimizer, cfg)

    # ------------------------------------------------
    # METRIC LEARNING SETUP
    # ------------------------------------------------
    miner = miners.BatchHardMiner()
    loss_fn = losses.TripletMarginLoss(margin=cfg.margin)

    # ------------------------------------------------
    # TRAINER OBJECT
    # ------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        miner=miner,
        cfg=cfg,
        scheduler=scheduler,
        oa_sched_dict=oa_sched_dict
    )

    trainer.train()


if __name__ == "__main__":
    main()