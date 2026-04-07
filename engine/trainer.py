import torch

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        cfg
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.cfg = cfg

        self.model.to(self.device)

    def train(self):

        for epoch in range(self.cfg.epochs):

            train_loss = self.train_epoch(epoch)

            print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

            if epoch % self.cfg.eval_freq == 0:
                self.evaluate(epoch)

    def train_epoch(self, epoch):

        self.model.train()

        total_loss = 0

        for imgs, labels in self.train_loader:

            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(imgs)

            loss = self.loss_fn(embeddings, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    def evaluate(self, epoch):

        self.model.eval()

        with torch.no_grad():

            for imgs, labels in self.val_loader:

                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(imgs)

                pass

        print(f"Evaluation finished for epoch {epoch}")