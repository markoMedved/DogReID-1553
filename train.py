from engine.trainer import Trainer
from data.dataloader import build_dataloader
from models.model_factory import build_model
from configs.config import load_config



def main():
    cfg = load_config()

    train_loader, val_loader = build_dataloader(cfg)

    model = build_model(cfg)

    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()

if __name__ == "__main__":
    main()