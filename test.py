import torch

from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer


def main():

    # Load configuration
    cfg = Config()

    print(f"Evaluating with model: {cfg.model}")

    # Build dataloaders
    # train_loader is unused but returned by the function
    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

    # Build pretrained model
    model = build_model(cfg)

    # Optimizer not needed for evaluation, but Trainer expects it
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        device=cfg.device,
        cfg=cfg
    )

    # Run evaluation
    rank1, rank5, mAP = trainer.evaluate()

    print("\nEvaluation Results")
    print(f"Rank-1: {rank1:.4f}")
    print(f"Rank-5: {rank5:.4f}")
    print(f"mAP: {mAP:.4f}")


if __name__ == "__main__":
    main()