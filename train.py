import torch
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from pytorch_metric_learning import losses, miners

def main():
    cfg = Config()
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(cfg.device)

    # --- 1. FREEZE EVERYTHING FIRST ---
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
    pool_layer = getattr(model, 'temporal_pool', getattr(model, 'temporal_attn', None))
    if pool_layer:
        for p in pool_layer.parameters(): p.requires_grad = True
    
    for p in model.bn.parameters(): 
        p.requires_grad = True
    
    # --- 4. DIFFERENTIAL LEARNING RATES ---
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and 'backbone' not in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr * 0.1}, 
        {"params": head_params, "lr": cfg.lr}
    ], weight_decay=cfg.weight_decay)

    # --- 5. BATCH-HARD MINING & TRIPLET LOSS ---
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