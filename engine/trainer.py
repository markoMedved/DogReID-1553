import os
import torch
from tqdm import tqdm
from pytorch_metric_learning.losses import TripletMarginLoss
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, device, cfg):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.optimizer = optimizer
        self.device = device
        self.cfg = cfg

        self.loss_fn = TripletMarginLoss(margin=0.3)

        os.makedirs(cfg.output_dir, exist_ok=True)
        #TODO remove
        rank1, rank5, mAP = self.evaluate()
        print(f"Eval -> Rank-1: {rank1:.4f} Rank-5: {rank5:.4f} mAP: {mAP:.4f}")

    def train(self):
        best_loss = float("inf")
        for epoch in range(self.cfg.epochs):
            loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} | Avg Loss: {loss:.4f}")

            if epoch % self.cfg.eval_period == 0:
                rank1, rank5, mAP = self.evaluate()
                print(f"Eval -> Rank-1: {rank1:.4f} Rank-5: {rank5:.4f} mAP: {mAP:.4f}")

            self.save_checkpoint(epoch, loss, "last_model.pth")
            if loss < best_loss:
                best_loss = loss
                self.save_checkpoint(epoch, loss, "best_model.pth")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for clips, labels, _, _ in pbar:
            clips = clips.to(self.device)   # (B, T, C, H, W)
            labels = labels.to(self.device) # (B)

            # 1. Forward Pass (Model now handles temporal pooling)
            embeddings = self.model(clips)  # (B, 768)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 2. Metric Learning Loss
            loss = self.loss_fn(embeddings, labels)

            # 3. Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def evaluate(self):
            self.model.eval()
            
            with torch.no_grad():
                def extract(loader, desc):
                    feats_list, ids_list = [], []
                    for clips, labels, _, _ in tqdm(loader, desc=desc):
                        clips = clips.to(self.device)
                        feat = self.model(clips) 
                        feat = F.normalize(feat, p=2, dim=1)
                        
                        feats_list.append(feat.cpu())
                        ids_list.extend(labels.tolist())
                    
                    return torch.cat(feats_list, dim=0), torch.tensor(ids_list)

                query_feats, query_ids = extract(self.query_loader, "Querying")
                gallery_feats, gallery_ids = extract(self.gallery_loader, "Gallerying")

            # Calculate Similarity
            sim_mat = query_feats @ gallery_feats.T
            
            # Apply the fix to ignore self-matches if they exist
            cmc, mAP = self.compute_metrics(sim_mat, query_ids, gallery_ids)
            return cmc[0], cmc[4], mAP

    def compute_metrics(self, similarity_mat, query_labels, gallery_labels):
        query_labels = torch.tensor(query_labels)
        gallery_labels = torch.tensor(gallery_labels)
        num_queries = len(query_labels)
        num_gallery = len(gallery_labels)

        cmc_curve = torch.zeros(num_gallery)
        ap_list = []

        for i in range(num_queries):
            # Sort gallery by similarity to current query
            sim_row = similarity_mat[i]
            sorted_idx = torch.argsort(sim_row, descending=True)
            
            # Binary mask of matches
            matches = (gallery_labels[sorted_idx] == query_labels[i]).float()
            
            if matches.sum() == 0: continue

            # Rank-N calculation
            rank = (matches == 1).nonzero(as_tuple=False)[0].item()
            cmc_curve[rank:] += 1

            # AP calculation
            cum_matches = matches.cumsum(0)
            precision = cum_matches / torch.arange(1, num_gallery + 1)
            ap = (precision * matches).sum() / matches.sum()
            ap_list.append(ap)

        return cmc_curve / num_queries, torch.tensor(ap_list).mean().item()

    def save_checkpoint(self, epoch, loss, filename):
        path = os.path.join(self.cfg.output_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss
        }, path)