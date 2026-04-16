import torch
from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer

def main():
    cfg = Config()

    print(f"--- Starting Pipeline ---")
    print(f"Model:  {cfg.model}")
    print(f"Device: {cfg.device}")
    print(f"World:  {cfg.world}")

    # 1. Setup Data
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)
    print(f"Data Loaded: {len(train_loader)} training batches.")

    # 2. Setup Model
    model = build_model(cfg)

    # 3. Setup Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        device=cfg.device,
        cfg=cfg
    )

    # 5. Launch
    if cfg.eval_only:
        print("Running Evaluation Only...")
        # Note: You might need to load weights here if eval_only is True
        rank1, rank5, mAP = trainer.evaluate()
        print(f"Rank-1: {rank1:.4f}, mAP: {mAP:.4f}")
    else:
        print("Starting Training...")
        trainer.train()

if __name__ == "__main__":
    main()