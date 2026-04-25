import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json

class Trainer:
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner):
        # move model to device (gpu/cpu)
        self.model = model.to(cfg.device)

        # dataloaders
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        self.optimizer = optimizer
        self.device = cfg.device
        self.cfg = cfg
        
        # metric learning components
        self.loss_fn = loss_fn
        self.miner = miner

        self.evaluate()

    def train(self):
        best_rank1 = 0.0
        val_split = getattr(self.cfg, 'val_split', 0)

        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            # run one full training epoch
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # validation experiment (dataset split contains validation identities)
            if val_split > 0:

                # evaluation every few epochs
                if (epoch + 1) % self.cfg.eval_period == 0:
                    rank1, rank5, mAP = self.evaluate()
                    
                    # keep best performing model
                    if rank1 > best_rank1:
                        best_rank1 = rank1
                        self.save_checkpoint("best_model.pth")
                
                # always keep last checkpoint
                self.save_checkpoint("last_model.pth")

        # final production training (no validation split)
        if val_split <= 0.01:
            print("!!! Final training run detected (val_split=0). Saving final model...")
            self.save_checkpoint("final_model.pth")

    def train_epoch(self, epoch):
        self.model.train()

        # gradient accumulation helps simulate larger batch sizes
        accum_steps = getattr(self.cfg, 'accum_steps', 8) 

        running_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for i, (videos, labels, dog_ids, video_ids) in enumerate(pbar):

            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # forward pass → embedding vectors
            embeddings = self.model(videos)
            
            # miner selects hardest positive/negative pairs in the batch
            hard_pairs = self.miner(embeddings, labels)

            # metric learning loss (triplet / margin based)
            loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            # divide loss for accumulation
            loss = loss / accum_steps
            loss.backward()

            # update weights only every accum_steps
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=loss.item() * accum_steps)

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        # switch model to inference mode
        self.model.eval()
        
        # extract normalized embeddings
        q_f, q_pids = self._get_features(self.query_loader, "Querying")
        g_f, g_pids = self._get_features(self.gallery_loader, "Gallerying")

        # closed world = every query identity exists in gallery
        if self.cfg.world == "closed":
            dist_mat = 1 - torch.mm(q_f, g_f.t())  # cosine distance
            r1, r5, mAP = self.calculate_cmc_map(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
            
            print(f"Eval (Closed) -> Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, mAP: {mAP:.2%}")
            return r1, r5, mAP

        # open world = some queries do not exist in gallery
        else:
            thresh, dir_curve, far_curve = self.dir_vs_far(q_f, q_pids, g_f, g_pids)
            
            # pick DIR values at specific FAR levels
            idx_1pct = np.argmin(np.abs(far_curve - 0.01))
            idx_5pct = np.argmin(np.abs(far_curve - 0.05))
            idx_10pct = np.argmin(np.abs(far_curve - 0.10))
            
            dir_1 = dir_curve[idx_1pct]
            dir_5 = dir_curve[idx_5pct]
            dir_10 = dir_curve[idx_10pct]

            print(f"Eval (Open) -> DIR@1%FAR: {dir_1:.2%}, DIR@5%FAR: {dir_5:.2%}, DIR@10%FAR: {dir_10:.2%}")
            
            # returned values used by training loop
            return dir_1, dir_5, dir_10

    def _get_features(self, loader, name):
        feats, pids = [], []

        for batch in tqdm(loader, desc=name):
            clips = batch[0].to(self.device)
            labels = batch[1]

            # extract embeddings
            f = self.model(clips)

            # normalize → cosine similarity becomes dot product
            f = F.normalize(f, p=2, dim=1)

            feats.append(f.cpu())
            pids.extend(labels.tolist())

        return torch.cat(feats, 0), torch.tensor(pids)

    def calculate_cmc_map(self, distmat, q_pids, g_pids):
        # classic person-reid evaluation
        num_q, num_g = distmat.shape

        # sort gallery by distance for each query
        indices = np.argsort(distmat, axis=1)

        # binary match matrix
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc, all_AP = [], []

        for i in range(num_q):

            row_matches = matches[i]

            # skip queries with no correct match
            if not np.any(row_matches):
                continue

            # first correct match position
            index = np.where(row_matches == 1)[0][0]
            all_cmc.append(index)

            # average precision computation
            cum_matches = np.cumsum(row_matches)
            prec = cum_matches / (np.arange(num_g) + 1)
            all_AP.append(np.sum(prec * row_matches) / np.sum(row_matches))

        cmc = np.zeros(num_g)

        for rank in all_cmc:
            cmc[rank:] += 1

        cmc /= len(all_cmc) if len(all_cmc) > 0 else 1

        return cmc[0], cmc[4], np.mean(all_AP)

    def dir_vs_far(self, query_features, query_labels, gallery_features, gallery_labels, thresholds=None):

        # used for open-world evaluation
        # DIR = detection identification rate
        # FAR = false accept rate

        if thresholds is None:
            thresholds = torch.linspace(0, 1, 500)
        
        q_f = F.normalize(query_features, p=2, dim=1)
        g_f = F.normalize(gallery_features, p=2, dim=1)

        similarity_mat = q_f @ g_f.T 

        q_labels = query_labels.to(q_f.device)
        g_labels = gallery_labels.to(g_f.device)

        match_matrix = q_labels[:, None] == g_labels[None, :]

        # queries that exist in gallery
        known_mask = torch.any(match_matrix, dim=1) 

        # queries that do not exist in gallery
        unknown_mask = ~known_mask

        known_sims = similarity_mat[known_mask]      
        known_matches = match_matrix[known_mask]     
        unknown_sims = similarity_mat[unknown_mask]  

        dir_list, far_list = [], []

        # best match per query
        max_vals_known, max_idx_known = known_sims.max(dim=1)

        # check if best match is correct identity
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

        # different folder depending on experiment vs final model
        val_split = getattr(self.cfg, 'val_split', 0)
        
        if val_split == 0:
            subfolder = f"final_model_{self.cfg.model}"
            target_dir = os.path.join(self.cfg.output_dir, subfolder)
        else:
            target_dir = self.cfg.output_dir
            
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        path = os.path.join(target_dir, filename)
        meta_path = path.replace(".pth", "_params.json")

        # handle DataParallel models
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint_data = {
            'model': state_dict,
            'epoch': getattr(self, 'current_epoch', 'unknown'),
            'val_split': val_split
        }

        torch.save(checkpoint_data, path)

        # parameters saved for reproducibility
        allowed_keys = ['lr', 'margin', 'weight_decay', 'batch_size', 'k', 'model', 'world', 'clip_len']

        params_to_save = {}

        for key in allowed_keys:

            if hasattr(self.cfg, key):
                params_to_save[key] = getattr(self.cfg, key)

            elif isinstance(self.cfg, dict) and key in self.cfg:
                params_to_save[key] = self.cfg[key]

        with open(meta_path, 'w') as f:
            json.dump(params_to_save, f, indent=4)
            
        print(f"Saved weights and metadata to: {target_dir}")