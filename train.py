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
    
    # Required for methodology argument parsing
    parser.add_argument('--reid_method', type=str, default='baseline', help='Which ReID method to use (e.g., bot)')
    parser.add_argument('--pooling_type', type=str, default='attention', help='Temporal pooling type')
    args = parser.parse_args()


    # ------------------------------------------------
    # LOAD DEFAULT CONFIG
    # ------------------------------------------------
    cfg = Config()

    if args.model: cfg.model = args.model
    if args.world: cfg.world = args.world
    if args.clip_len: cfg.clip_len = args.clip_len
    if args.lr: cfg.lr = args.lr
    if args.margin: cfg.margin = args.margin
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.k: cfg.k = args.k
    
    # Pass new args to config
    cfg.reid_method = args.reid_method
    cfg.pooling_type = args.pooling_type
    
    # Ensure backbone exists for naming
    backbone_name = getattr(cfg, 'backbone', cfg.model)
    full_ft = getattr(cfg, 'full_finetune', True)

    # --- UPDATED RUN NAME (Methodology Sec 3.1) ---
    cfg.run_name = f"{backbone_name}_{cfg.reid_method}_{cfg.world}_{cfg.pooling_type}_finetune_{full_ft}"
    cfg.output_dir = cfg.project_root / "trained_models" / cfg.run_name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

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
    # We still use custom parameter groups, but rely entirely on apply_freezing's output
    backbone_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'backbone' in n
    ]

    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and 'backbone' not in n
    ]

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