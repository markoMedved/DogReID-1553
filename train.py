import torch
import argparse
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners

def main():
# --- 0. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Dog Re-ID Training")
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--margin', type=float, default=None, help='Triplet loss margin')
    parser.add_argument('--weight_decay', type=float, default=None, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (P*K)')
    parser.add_argument('--k', type=int, default=None, help='Clips per dog ID')
    parser.add_argument('--model', type=str, default=None, help="Backbone: 'dinov2', 'swin', 'vit'")
    parser.add_argument('--world', type=str, default=None, help="'closed' or 'open'")
    parser.add_argument('--clip_len', type=int, default=None, help='Frames per video clip')

    args = parser.parse_args()

    # Initialize default config
    cfg = Config()

    # --- 1. OVERRIDE CONFIG WITH CLI ARGS ---
    # Only override if the user actually provided the flag
    if args.model: cfg.model = args.model
    if args.world: cfg.world = args.world
    if args.clip_len: cfg.clip_len = args.clip_len
    if args.lr: cfg.lr = args.lr
    if args.margin: cfg.margin = args.margin
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    
    # Special handling for Batch Size and K
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.k: cfg.k = args.k

    # Re-display config to confirm overrides
    cfg.display()

    # --- 2. REST OF YOUR PIPELINE ---
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(cfg.device)

    # ... [Keep your unfreezing logic exactly as it was] ...
    for p in model.parameters():
        p.requires_grad = False
    
    # Handle DINOv2 / Swin / ViT unfreezing (omitted for brevity, keep yours!)
    # (Insert your Architecture-Aware unfreezing block here)

    # Always unfreeze the head
    pool_layer = getattr(model, 'temporal_pool', getattr(model, 'temporal_attn', None))
    if pool_layer:
        for p in pool_layer.parameters(): p.requires_grad = True
    for p in model.bn.parameters(): 
        p.requires_grad = True
    
    # --- 3. OPTIMIZER WITH OVERRIDDEN VALUES ---
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1}, 
        {"params": head_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # --- 4. LOSS & TRAINER ---
    miner = miners.BatchHardMiner()
    loss_fn = losses.TripletMarginLoss(margin=cfg.margin)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,     
        gallery_loader=gallery_loader, 
        optimizer=optimizer,
        loss_fn=loss_fn,
        miner=miner,
        cfg=cfg
    )
    
    trainer.train()

if __name__ == "__main__":
    main()