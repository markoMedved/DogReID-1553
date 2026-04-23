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
    if args.model: cfg.model = args.model
    if args.world: cfg.world = args.world
    if args.clip_len: cfg.clip_len = args.clip_len
    if args.lr: cfg.lr = args.lr
    if args.margin: cfg.margin = args.margin
    if args.weight_decay: cfg.weight_decay = args.weight_decay
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.k: cfg.k = args.k

    cfg.display()

    # --- 2. DATA & MODEL ---
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(cfg.device)

    # --- 3. ARCHITECTURE-AWARE UNFREEZING ---
    # First, freeze everything
    for p in model.parameters():
        p.requires_grad = False
    
    # --- 2. ARCHITECTURE-AWARE PARTIAL UNFREEZING ---
    # Handle Torchvision ViT (vit_b_16 / vit_l_16)
    if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layers'):
        for layer in model.backbone.encoder.layers[-2:]:
            for p in layer.parameters(): p.requires_grad = True
        if hasattr(model.backbone.encoder, 'ln'):
            for p in model.backbone.encoder.ln.parameters(): p.requires_grad = True

    # Handle DINOv2 (flat blocks structure)
    elif hasattr(model.backbone, 'blocks'):
        for block in model.backbone.blocks[-2:]:
            for p in block.parameters(): p.requires_grad = True
        if hasattr(model.backbone, 'norm'):
            for p in model.backbone.norm.parameters(): p.requires_grad = True

    # Handle Swin (hierarchical layers structure)
    elif hasattr(model.backbone, 'layers'):
        for p in model.backbone.layers[-1].parameters(): p.requires_grad = True
        if hasattr(model.backbone, 'norm'):
            for p in model.backbone.norm.parameters(): p.requires_grad = True

    # --- 3. ALWAYS UNFREEZE THE HEAD ---
    # Note: Using your internal naming 'temporal_pool' or 'temporal_attn'
    # Always unfreeze the head
    pool_layer = getattr(model, 'temporal_pool', getattr(model, 'temporal_attn', None))
    if pool_layer:
        for p in pool_layer.parameters(): p.requires_grad = True
    
    for p in model.bn.parameters(): 
        p.requires_grad = True

    # --- 4. OPTIMIZER ---
    # Differential Learning Rates: Backbone (Lower) vs Head (Higher)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1}, 
        {"params": head_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # --- 5. LOSS & TRAINER ---
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