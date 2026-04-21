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

        self.evaluate()

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
        
        q_f, q_pids = self._get_features(self.query_loader, "Querying")
        g_f, g_pids = self._get_features(self.gallery_loader, "Gallerying")

        # --- CLOSED WORLD METRICS (Always calculated as baseline) ---
        dist_mat = 1 - torch.mm(q_f, g_f.t())
        
        
        if self.cfg.world == "closed":
            r1, r5, mAP = self.calculate_cmc_map(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
            print(f"Eval (Closed) -> Rank-1: {r1:.2%}, mAP: {mAP:.2%}")
            return r1, mAP

        # --- OPEN WORLD METRICS ---
        else:
            thresh, dir_curve, far_curve = self.dir_vs_far(q_f, q_pids, g_f, g_pids)
            
            # Calculate at 3 levels: 0.1%, 1%, 10%
            far_levels = [0.01, 0.05, 0.1]
            dir_at_far = {}
            
            print(f"Eval (Open) -> Rank-1: {r1:.2%}, mAP: {mAP:.2%}")
            for level in far_levels:
                idx = np.argmin(np.abs(far_curve - level))
                val = dir_curve[idx]
                dir_at_far[level] = val
                print(f"  DIR @ {level*100}% FAR: {val:.2%}")
            
            return r1, mAP, dir_at_far

    def _get_features(self, loader, name):
        feats, pids = [], []
        for batch in tqdm(loader, desc=name):
            clips = batch[0].to(self.device)
            labels = batch[1]
            f = self.model(clips)
            # Normalize here ensures mm is Cosine Similarity
            f = F.normalize(f, p=2, dim=1)
            feats.append(f.cpu())
            pids.extend(labels.tolist())
        return torch.cat(feats, 0), torch.tensor(pids)

    def calculate_cmc_map(self, distmat, q_pids, g_pids):
        num_q, num_g = distmat.shape
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc, all_AP = [], []
        for i in range(num_q):
            row_matches = matches[i]
            if not np.any(row_matches): continue
            index = np.where(row_matches == 1)[0][0]
            all_cmc.append(index)
            cum_matches = np.cumsum(row_matches)
            prec = cum_matches / (np.arange(num_g) + 1)
            all_AP.append(np.sum(prec * row_matches) / np.sum(row_matches))

        cmc = np.zeros(num_g)
        for rank in all_cmc:
            cmc[rank:] += 1
        cmc /= len(all_cmc) if len(all_cmc) > 0 else 1
        return cmc[0], cmc[4], np.mean(all_AP)

    def dir_vs_far(self, query_features, query_labels, gallery_features, gallery_labels, thresholds=None):
        # ... (Your logic remains exactly the same, integrated as class method)
        if thresholds is None:
            thresholds = torch.linspace(0, 1, 500)
        
        q_f = F.normalize(query_features, p=2, dim=1)
        g_f = F.normalize(gallery_features, p=2, dim=1)

        similarity_mat = q_f @ g_f.T 
        q_labels = query_labels.to(q_f.device)
        g_labels = gallery_labels.to(g_f.device)
        match_matrix = q_labels[:, None] == g_labels[None, :]

        known_mask = torch.any(match_matrix, dim=1) 
        unknown_mask = ~known_mask

        known_sims = similarity_mat[known_mask]      
        known_matches = match_matrix[known_mask]     
        unknown_sims = similarity_mat[unknown_mask]  

        dir_list, far_list = [], []
        max_vals_known, max_idx_known = known_sims.max(dim=1)
        top_is_correct = known_matches.gather(1, max_idx_known.unsqueeze(1)).squeeze(1)
        max_vals_unknown, _ = unknown_sims.max(dim=1) if unknown_sims.numel() > 0 else (torch.tensor([]), None)

        for thresh in thresholds:
            if known_sims.numel() > 0:
                dir_val = ((max_vals_known > thresh) & top_is_correct).float().mean().item()
            else:
                dir_val = 0.0
            dir_list.append(dir_val)

            if unknown_sims.numel() > 0:
                far_val = (max_vals_unknown > thresh).float().mean().item()
            else:
                far_val = 0.0
            far_list.append(far_val)

        return thresholds.cpu().numpy(), np.array(dir_list), np.array(far_list)

    def save_checkpoint(self, filename):
        path = os.path.join(self.cfg.output_dir, filename)
        torch.save(self.model.state_dict(), path)