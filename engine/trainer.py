import torch
import os
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        loss_fn,
        device,
        cfg
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg

        self.model.to(device)

        os.makedirs(cfg.output_dir, exist_ok=True)

    def train(self):

        best_loss = float("inf")

        for epoch in range(self.cfg.epochs):

            loss = self.train_epoch(epoch)

            print(f"\nEpoch {epoch} | Avg Loss: {loss:.4f}")

            self.save_checkpoint(epoch, loss, "last_model.pth")

            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint(epoch, loss, "best_model.pth")

    def train_epoch(self, epoch):

        self.model.train()

        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for clips, labels, _, _ in pbar:

            clips = clips.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(clips)

            loss = self.loss_fn(embeddings, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value

            pbar.set_postfix(loss=loss_value)

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def save_checkpoint(self, epoch, loss, filename):

        path = os.path.join(self.cfg.output_dir, filename)

        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }, path)