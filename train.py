import torch
import argparse
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners

# --- NEW IMPORTS (Methodology Sec 3.4) ---
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
    parser.add_argument('--model', type=str, default=None, help="Backbone: 'dinov2', 'swin', 'vit'")
    parser.add_argument('--world', type=str, default=None, help="'closed' or 'open'")
    parser.add_argument('--clip_len', type=int, default=None, help='Frames per video clip')
    
    # Re-ID methodology. Defaults are None so that configs/config.py stays
    # authoritative unless a flag is explicitly passed.
    parser.add_argument('--reid_method', type=str, default=None,
                        choices=['bot', 'transreid', 'baseline'],
                        help="Re-ID method; 'baseline' uses the legacy builders")
    parser.add_argument('--pooling_type', type=str, default=None,
                        choices=['attention', 'mean', 'max'], help='Temporal pooling type')
    parser.add_argument('--full_finetune', dest='full_finetune', action='store_true', default=None,
                        help='Train the whole backbone')
    parser.add_argument('--no_full_finetune', dest='full_finetune', action='store_false',
                        help='Train only the last unfreeze_blocks backbone blocks')
    parser.add_argument('--unfreeze_blocks', type=int, default=None,
                        help='Trailing backbone blocks to train when not full fine-tuning')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    args = parser.parse_args()


    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()

    # --- Command-Line Overrides ---
    # Only applied when the flag was actually passed, so config.py stays
    # authoritative otherwise. '--model' sets the backbone too, or VideoReID
    # would keep building whatever cfg.backbone says.
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
    if args.reid_method is not None: cfg.reid_method = args.reid_method
    if args.pooling_type is not None: cfg.pooling_type = args.pooling_type
    if args.full_finetune is not None: cfg.full_finetune = args.full_finetune
    if args.unfreeze_blocks is not None: cfg.unfreeze_blocks = args.unfreeze_blocks

    # --- Schedule Sanity Check ---
    # Milestones outside the run length mean the learning rate never decays
    if not any(m < cfg.epochs for m in cfg.lr_milestones):
        raise ValueError(
            f"lr_milestones {cfg.lr_milestones} all fall outside epochs={cfg.epochs}; "
            f"the learning rate would never decay. Rescale them together."
        )

    # --- Recompute Derived Fields ---
    # run_name, output_dir and num_ids all depend on the values above
    cfg.refresh_run_name()

    cfg.display()


    # ------------------------------------------------
    # BUILD DATA LOADERS
    # ------------------------------------------------
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)


    # --- DYNAMIC NUM_CLASSES (Methodology Sec 3.4) ---
    # train_loader.dataset is a Subset, so id_map lives one level deeper
    cfg.num_classes = len(train_loader.dataset.dataset.id_map)
    print(f"[cfg] num_classes = {cfg.num_classes}")


    # ------------------------------------------------
    # BUILD MODEL
    # ------------------------------------------------
    model = build_model(cfg).to(cfg.device)


    # --- DELEGATE FREEZING (Methodology Sec 3.4) ---
    # This replaces all the manual requires_grad loops and prevents the model.bn crash
    model = apply_freezing(model, cfg)
    print(trainable_report(model))


    # ------------------------------------------------
    # OPTIMIZER
    # ------------------------------------------------
    # Two parameter groups, split by initialization rather than by module name.
    # Pretrained weights get the reduced learning rate; randomly initialized
    # heads get the full one. TransReID's JPM local branch is a copy of a
    # pretrained transformer block, so it belongs with the backbone even though
    # it is not named 'backbone'.
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

    n_bb = sum(p.numel() for p in backbone_params)
    n_hd = sum(p.numel() for p in head_params)
    print(f"[optim] pretrained {n_bb:,} params @ lr*0.1 | heads {n_hd:,} params @ lr")
    assert n_bb + n_hd == sum(p.numel() for p in model.parameters() if p.requires_grad), \
        "parameter groups do not cover every trainable parameter"

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr}
        ],
        weight_decay=cfg.weight_decay
    )

    # --- BUILD SCHEDULER (Methodology Sec 3.4) ---
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
        scheduler=scheduler  # Added per doc
    )

    trainer.train()

if __name__ == "__main__":
    main()