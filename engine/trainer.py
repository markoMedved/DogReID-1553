import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json

class Trainer:
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner):
        # --- Move Model to Compute Device ---
        self.model = model.to(cfg.device)

        # --- Dataloaders ---
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        # --- Training Configuration ---
        self.optimizer = optimizer
        self.device = cfg.device
        self.cfg = cfg
        
        # --- Metric Learning Components ---
        self.loss_fn = loss_fn
        self.miner = miner

        # TODO remove
        self.evaluate()

    def train(self):
        # --- Initialize Tracking Variables ---
        best_rank1 = 0.0
        val_split = getattr(self.cfg, 'val_split', 0)

        # --- Main Training Loop ---
        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            # Run one full training epoch
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            # --- Validation Evaluation ---
            if val_split > 0:
                # Run evaluation only at specified intervals
                if (epoch + 1) % self.cfg.eval_period == 0:
                    rank1, rank5, mAP = self.evaluate()
                    

        # --- Final Model Saving ---
        # Automatically saves the model if trained on the full dataset
        if val_split <= 0.01:
            print("!!! Final training run detected (val_split=0). Saving final model...")
            self.save_checkpoint("model.pth")

    def train_epoch(self, epoch):
        # --- Setup Training Epoch ---
        self.model.train()

        # Gradient accumulation helps simulate larger batch sizes on constrained hardware
        accum_steps = getattr(self.cfg, 'accum_steps', 8) 

        running_loss = 0.0
        self.optimizer.zero_grad()

        # Initialize progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        # --- Batch Processing Loop ---
        for i, (videos, labels, dog_ids, video_ids) in enumerate(pbar):

            # Move data to the active device
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Forward Pass -> Generate embedding vectors
            embeddings = self.model(videos)
            
            # --- Hard Pair Mining ---
            # Selects the hardest positive/negative pairs to optimize learning
            hard_pairs = self.miner(embeddings, labels)

            # --- Metric Learning Loss ---
            # Computes triplet or margin-based loss using the mined pairs
            loss = self.loss_fn(embeddings, labels, hard_pairs)
            
            # --- Backpropagation with Accumulation ---
            # Divides loss by accumulation steps to average gradients correctly
            loss = loss / accum_steps
            loss.backward()

            # Update weights only after specified accumulation steps
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # --- Update Progress Logging ---
            running_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=loss.item() * accum_steps)

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        # --- Switch Model to Inference Mode ---
        self.model.eval()
        
        # --- Feature Extraction ---
        # Extracts normalized embeddings for both query and gallery sets
        q_f, q_pids = self._get_features(self.query_loader, "Querying")
        g_f, g_pids = self._get_features(self.gallery_loader, "Gallerying")

        # --- Closed-World Evaluation ---
        # Assumes every query identity exists within the gallery
        if self.cfg.world == "closed":
            dist_mat = 1 - torch.mm(q_f, g_f.t())  # Calculate cosine distance
            r1, r5, mAP = self.calculate_cmc_map(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
            
            print(f"Eval (Closed) -> Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, mAP: {mAP:.2%}")
            return r1, r5, mAP

        # --- Open-World Evaluation ---
        # Assumes some queries may not exist within the gallery
        else:
            thresh, dir_curve, far_curve = self.dir_vs_far(q_f, q_pids, g_f, g_pids)
            
            # Select DIR values at specific False Accept Rates (FAR)
            idx_1pct = np.argmin(np.abs(far_curve - 0.01))
            idx_5pct = np.argmin(np.abs(far_curve - 0.05))
            idx_10pct = np.argmin(np.abs(far_curve - 0.10))
            
            dir_1 = dir_curve[idx_1pct]
            dir_5 = dir_curve[idx_5pct]
            dir_10 = dir_curve[idx_10pct]

            print(f"Eval (Open) -> DIR@1%FAR: {dir_1:.2%}, DIR@5%FAR: {dir_5:.2%}, DIR@10%FAR: {dir_10:.2%}")
            
            # Returned values are typically logged by the training loop
            return dir_1, dir_5, dir_10

    def _get_features(self, loader, name):
        # --- Extract Embeddings from Dataloader ---
        feats, pids = [], []

        for batch in tqdm(loader, desc=name):
            clips = batch[0].to(self.device)
            labels = batch[1]

            # Forward pass to get features
            f = self.model(clips)

            # Normalize embeddings -> Allows cosine similarity via dot product
            f = F.normalize(f, p=2, dim=1)

            feats.append(f.cpu())
            pids.extend(labels.tolist())

        return torch.cat(feats, 0), torch.tensor(pids)

    def calculate_cmc_map(self, distmat, q_pids, g_pids):
        # --- Classic Person/Object Re-ID Evaluation Metrics ---
        num_q, num_g = distmat.shape

        # Sort the gallery indices by distance for each query
        indices = np.argsort(distmat, axis=1)

        # Create a binary matrix indicating true matches
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        all_cmc, all_AP = [], []

        for i in range(num_q):

            row_matches = matches[i]

            # Skip calculation if the query has no correct match in the gallery
            if not np.any(row_matches):
                continue

            # Find the position of the first correct match
            index = np.where(row_matches == 1)[0][0]
            all_cmc.append(index)

            # --- Average Precision (AP) Computation ---
            cum_matches = np.cumsum(row_matches)
            prec = cum_matches / (np.arange(num_g) + 1)
            all_AP.append(np.sum(prec * row_matches) / np.sum(row_matches))

        cmc = np.zeros(num_g)

        # Accumulate counts for CMC curve
        for rank in all_cmc:
            cmc[rank:] += 1

        # Normalize to get probabilities
        cmc /= len(all_cmc) if len(all_cmc) > 0 else 1

        return cmc[0], cmc[4], np.mean(all_AP)

    def dir_vs_far(self, query_features, query_labels, gallery_features, gallery_labels, thresholds=None):
            # --- Open-World Evaluation Metrics ---
            # DIR = Detection Identification Rate (True Positive Rate)
            # FAR = False Accept Rate (False Positive Rate)

            # Ensure features are normalized
            q_f = F.normalize(query_features, p=2, dim=1)
            g_f = F.normalize(gallery_features, p=2, dim=1)

            # Calculate Distance Matrix via Euclidean Distance (matches your numpy logic)
            # L2 normalized features have a maximum Euclidean distance of 2.0
            dist_mat = torch.cdist(q_f, g_f)

            q_labels = query_labels.to(q_f.device)
            g_labels = gallery_labels.to(g_f.device)

            # Matrix indicating which queries match which gallery images
            match_matrix = q_labels[:, None] == g_labels[None, :]

            # Masks separating known (in gallery) vs unknown (not in gallery) queries
            known_mask = torch.any(match_matrix, dim=1) 
            unknown_mask = ~known_mask

            # Find the smallest distance (best match) and its index
            best_dist, best_idx = torch.min(dist_mat, dim=1)
            correct_match = g_labels[best_idx] == q_labels

            if thresholds is None:
                thresholds = torch.linspace(0, 2, 10000, device=q_f.device)
            else:
                thresholds = thresholds.to(q_f.device)

            # Filter distances and matches based on masks
            known_dists = best_dist[known_mask]
            known_correct = correct_match[known_mask]
            unknown_dists = best_dist[unknown_mask]

            n_known = known_mask.sum().item()
            n_unknown = unknown_mask.sum().item()

            if n_known > 0:
                under_and_correct = (known_dists[:, None] <= thresholds) & known_correct[:, None]
                dir_array = under_and_correct.float().sum(dim=0) / n_known
            else:
                dir_array = torch.zeros_like(thresholds)

            # Vectorized FAR calculation: Unknown dog distance falls below threshold
            if n_unknown > 0:
                under_unknown = unknown_dists[:, None] <= thresholds
                far_array = under_unknown.float().sum(dim=0) / n_unknown
            else:
                far_array = torch.zeros_like(thresholds)

            # Return identical format to your original dir_vs_far return statement
            return thresholds.cpu().numpy(), dir_array.cpu().numpy(), far_array.cpu().numpy()


    def save_checkpoint(self, filename):

        # --- Directory Configuration ---
        val_split = getattr(self.cfg, 'val_split', 0)
        target_dir = self.cfg.output_dir
            
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)

        path = os.path.join(target_dir, filename)
        meta_path = path.replace(".pth", "_params.json")

        # --- Model State Extraction ---
        # Handle models wrapped in DataParallel
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()

        checkpoint_data = {
            'model': state_dict,
            'epoch': getattr(self, 'current_epoch', 'unknown'),
            'val_split': val_split
        }

        # Save weights
        torch.save(checkpoint_data, path)

        # --- Metadata Saving ---
        # Save specific config parameters alongside the model for reproducibility
        allowed_keys = ['lr', 'margin', 'weight_decay', 'batch_size', 
                        'k', 'model', 'world', 'clip_len', 'epochs',
                        "accum_steps", "num_workers", "chunk_size"]

        params_to_save = {}

        for key in allowed_keys:

            if hasattr(self.cfg, key):
                params_to_save[key] = getattr(self.cfg, key)

            elif isinstance(self.cfg, dict) and key in self.cfg:
                params_to_save[key] = self.cfg[key]

        with open(meta_path, 'w') as f:
            json.dump(params_to_save, f, indent=4)
            
        print(f"Saved weights and metadata to: {target_dir}")