import torch

from configs.config import Config
from data.dataloader import build_dataloaders
from models.model_factory import build_model
from engine.trainer import Trainer
from losses.triplet_loss import TripletLoss


def main():

    cfg = Config()

    print(f"Training with model: {cfg.model}")

    train_loader, query_loader, gallery_loader = build_dataloaders(cfg)

    model = build_model(cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    loss_fn = TripletLoss(margin=0.3)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=cfg.device,
        cfg=cfg
    )

    trainer.train()


if __name__ == "__main__":
    main()