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

    args = parser.parse_args()

    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()

    # --- Command-Line Overrides ---
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
    if args.reid_method is not None:
        cfg.reid_method = None if args.reid_method == 'baseline' else args.reid_method
    if args.pooling_type is not None: cfg.pooling_type = args.pooling_type
    if args.full_finetune is not None: cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None: cfg.unfreeze_blocks = args.unfreeze_blocks
    if args.use_id_loss is not None: cfg.use_id_loss = args.use_id_loss

    # Re-ID BoT hyperparameter adjustments if reid_method is active and not overridden
    if cfg.reid_method in ('bot', 'transreid'):
        if args.epochs is None: cfg.epochs = 51
        if args.lr is None: cfg.lr = 2e-05
        if args.clip_len is None: cfg.clip_len = 8
        cfg.accum_steps = 2

        if not any(m < cfg.epochs for m in cfg.lr_milestones):
            raise ValueError(
                f"lr_milestones {cfg.lr_milestones} all fall outside epochs={cfg.epochs}; "
                f"the learning rate would never decay. Rescale them together."
            )

    cfg.update_model_settings()
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
    # OPTIMIZER
    # ------------------------------------------------
    def is_pretrained(name):
        return name.startswith('backbone') or name.startswith('jpm.')

    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and is_pretrained(n)
    ]

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not is_pretrained(n)
    ]

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.lr * 0.1})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr})

    if not param_groups:
        raise ValueError(
            "No trainable parameters found for optimizer! "
            "Ensure at least part of the backbone or head has requires_grad=True."
        )

    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[optim] pretrained {n_bb:,} params @ lr*0.1 | heads {n_hd:,} params @ lr")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    # --- BUILD SCHEDULER ---
    scheduler = None
    if cfg.reid_method in ('bot', 'transreid'):
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
        scheduler=scheduler
    )

    trainer.train()


if __name__ == "__main__":
    main()