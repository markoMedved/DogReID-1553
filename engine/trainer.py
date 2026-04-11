import torch
import os
from tqdm import tqdm
import numpy as np


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        query_loader,
        gallery_loader,
        optimizer,
        loss_fn,
        device,
        cfg
    ):
        self.model = model
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

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

            rank1, rank5, mAP = self.evaluate()

            print(
                f"Evaluation → "
                f"Rank-1: {rank1:.4f} "
                f"Rank-5: {rank5:.4f} "
                f"mAP: {mAP:.4f}"
            )

            self.save_checkpoint(epoch, loss, "last_model.pth")

            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint(epoch, loss, "best_model.pth")

    def train_epoch(self, epoch):

        self.model.train()

        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (clips, labels, dog_ids, video_names) in enumerate(pbar):


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

    def evaluate(self):

        self.model.eval()

        query_feats = []
        gallery_feats = []

        query_ids = []
        gallery_ids = []

        with torch.no_grad():

            # ----- QUERY -----
            for clips, labels, dog_ids, video_names in self.query_loader:

                clips = clips.to(self.device)

                feats = self.model(clips)

                query_feats.append(feats.cpu())
                query_ids.extend(labels.numpy())

            # ----- GALLERY -----
            for clips, labels, dog_ids, video_names in self.gallery_loader:

                clips = clips.to(self.device)

                feats = self.model(clips)

                gallery_feats.append(feats.cpu())
                gallery_ids.extend(labels.numpy())

        query_feats = torch.cat(query_feats)
        gallery_feats = torch.cat(gallery_feats)

        distmat = self.compute_distance_matrix(query_feats, gallery_feats)

        cmc, mAP = self.compute_metrics(distmat, query_ids, gallery_ids)

        return cmc[0], cmc[4], mAP
    
def compute_distance_matrix(self, qf, gf):

    qf = torch.nn.functional.normalize(qf, dim=1)
    gf = torch.nn.functional.normalize(gf, dim=1)

    dist = 1 - torch.mm(qf, gf.t())

    return dist.cpu().numpy()


def compute_metrics(self, distmat, q_ids, g_ids):

    q_ids = np.asarray(q_ids)
    g_ids = np.asarray(g_ids)

    indices = np.argsort(distmat, axis=1)
    matches = (g_ids[indices] == q_ids[:, None])

    all_cmc = []
    all_AP = []

    for i in range(len(q_ids)):

        match = matches[i]

        if not np.any(match):
            continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc)

        num_rel = match.sum()

        tmp_cmc = match.cumsum()
        precision = tmp_cmc / (np.arange(len(match)) + 1)

        AP = (precision * match).sum() / num_rel
        all_AP.append(AP)

    cmc = np.mean(all_cmc, axis=0)
    mAP = np.mean(all_AP)

    return cmc, mAP