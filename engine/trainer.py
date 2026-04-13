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

            if epoch % self.cfg.eval_period == 0:
                

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
    
    def compute_similarity_matrix(self, qf, gf):

        qf = torch.nn.functional.normalize(qf, dim=1)
        gf = torch.nn.functional.normalize(gf, dim=1)

        sim_mat = qf @ gf.T

        return sim_mat.cpu().numpy()


    def compute_metrics(self, similarity_mat, query_labels, gallery_labels):
        # Number of query and gallery feature vectors
        num_queris =  len(query_labels)
        num_gallery = len(gallery_labels)

        q_ids = np.asarray(q_ids)
        g_ids = np.asarray(g_ids)

        # Average precision list
        ap_list = []

        for query_label, similarity_row in zip(query_labels, similarity_mat):

            # Get the sorted indices by similarity for the current row
            sorted_indices = torch.argsort(similarity_row, descending=True)

            # Find where the gallery labels match the query label
            matches = (gallery_labels[sorted_indices] == query_label).float()
            num_rel = matches.sum()

            # CMC
            # Rank is the lowest non-zero index in the matches
            rank = matches.nonzero(as_tuple=False)[0].item()
            cmc_curve[rank:] += 1

            # Average precision
            # Cumulative matches up to each rank
            cum_matches = matches.cumsum(0)
            # Devide each sum with the rank - precision at k
            precision_at_k = cum_matches / (torch.arange(1, num_gallery + 1).float())
            # Average it
            ap = (precision_at_k * matches).sum() / num_rel
            ap_list.append(ap)

        cmc_curve = cmc_curve / num_queris
        mAP = sum(ap_list) / len(ap_list)
        return cmc_curve, mAP