import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.optimizer = optimizer
        self.device = cfg.device
        self.cfg = cfg
        
        # Now these match your main.py arguments
        self.loss_fn = loss_fn
        self.miner = miner

    def train(self):
            best_rank1 = 0.0
            for epoch in range(self.cfg.epochs):
                avg_loss = self.train_epoch(epoch)
                print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

                if (epoch + 1) % self.cfg.eval_period == 0:
                    rank1, rank5, mAP = self.evaluate()
                    print(f"Eval -> Rank-1: {rank1:.2%}, Rank-5: {rank5:.2%}, mAP: {mAP:.2%}")

                    if rank1 > best_rank1:
                        best_rank1 = rank1
                        self.save_checkpoint("best_model.pth")
                
                # 4. Save last (Indented 8 spaces/2 tabs)
                self.save_checkpoint("last_model.pth")

    def train_epoch(self, epoch):
        self.model.train()
        accum_steps = getattr(self.cfg, 'accum_steps', 8) 
        running_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        # Use underscores (_) for values you don't need during training
        for i, (videos, labels, dog_ids, video_ids) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # 1. Forward pass
            embeddings = self.model(videos)
            
            # 2. Mining & Loss
            # BatchHardMiner finds the most difficult triplets in the current batch
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            # 3. Gradient Accumulation
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=loss.item() * accum_steps)

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        def get_features(loader, name):
            feats, pids = [], []
            for batch in tqdm(loader, desc=name):
                # Using batch[0] and batch[1] is safer than unpacking if metadata varies
                clips = batch[0].to(self.device)
                labels = batch[1]
                
                f = self.model(clips)
                feats.append(f.cpu())
                pids.extend(labels.tolist())
            return torch.cat(feats, 0), torch.tensor(pids)

        q_f, q_pids = get_features(self.query_loader, "Querying")
        g_f, g_pids = get_features(self.gallery_loader, "Gallerying")

        # Fast Cosine Similarity
        dist_mat = 1 - torch.mm(q_f, g_f.t())
        
        rank1, rank5, mAP = self.calculate_metrics(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
        return rank1, rank5, mAP

    def calculate_metrics(self, distmat, q_pids, g_pids):
        """Standard Rank-N and mAP metrics for Re-ID"""
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc = []
        all_AP = []
        for i in range(num_q):
            row_matches = matches[i]
            if not np.any(row_matches): continue
            
            # Rank-1 index
            index = np.where(row_matches == 1)[0][0]
            all_cmc.append(index)
            
            # Average Precision
            cum_matches = np.cumsum(row_matches)
            prec = cum_matches / (np.arange(num_g) + 1)
            all_AP.append(np.sum(prec * row_matches) / np.sum(row_matches))

        cmc = np.zeros(num_g)
        for rank in all_cmc:
            cmc[rank:] += 1
        cmc /= len(all_cmc)

        return cmc[0], cmc[4], np.mean(all_AP)

    def save_checkpoint(self, filename):
        path = os.path.join(self.cfg.output_dir, filename)
        torch.save(self.model.state_dict(), path)