import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json

class Trainer:
    "Class that has the training logic"
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner,
                 scheduler=None):
        # --- Move Model to Compute Device ---
        self.model = model.to(cfg.device)

        # --- Dataloaders ---
        self.train_loader = train_loader
        self.query_loader = query_loader # Validation
        self.gallery_loader = gallery_loader # Validation

        # --- Training Configuration ---
        self.optimizer = optimizer
        self.scheduler = scheduler # Stepped once per epoch, may be None
        self.device = cfg.device
        self.cfg = cfg

        # --- Metric Learning Components ---
        self.loss_fn = loss_fn
        self.miner = miner

        # --- Identity Loss ---
        # Used only when the model exposes a classifier (cfg.num_classes > 0)
        self.id_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.id_loss_weight = getattr(cfg, 'id_loss_weight', 1.0)

        # --- Mixed Precision ---
        # Only active on CUDA. bfloat16 needs no gradient scaler; float16 does.
        amp = getattr(cfg, 'amp', 'bf16')
        self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(amp)
        self.amp_enabled = self.amp_dtype is not None and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )
        if self.amp_enabled:
            print(f"[amp] mixed precision enabled ({amp})")


    def train(self):
        """Train the model"""
        # --- Get the validation split ratio ---
        val_split = getattr(self.cfg, 'val_split', 0)

        # --- Main Training Loop ---
        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch

            # Run one full training epoch
            avg_loss = self.train_epoch(epoch)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

            # --- Learning Rate Schedule ---
            # Stepped per epoch, matching the warmup and milestone units
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Validation Evaluation ---
            if val_split > 0 and self.cfg.world == "closed":
                # Run evaluation only at specified intervals
                if (epoch + 1) % self.cfg.eval_period == 0:
                    rank1, rank5, mAP = self.evaluate()
                    

        # --- Final Model Saving ---
        # Automatically saves the model if trained on the full dataset
        if val_split <= 0.01:
            print("!!! Final training run detected (val_split=0). Saving final model...")
            self.save_checkpoint("model.pth")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        # --- Put into train mode ---
        self.model.train()

        # Gradient accumulation helps simulate larger batch sizes
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
            # Models with an identity head return (embeddings, logits)
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.amp_enabled):
                outputs = self.model(videos)

            if isinstance(outputs, tuple):
                embeddings, logits = outputs
            else:
                embeddings, logits = outputs, None

            # Losses are computed in float32; metric learning is sensitive to
            # reduced precision in the distance matrix
            embeddings = embeddings.float()
            if logits is not None:
                logits = logits.float()

            # --- Hard Pair Mining ---
            # Selects the hardest positive/negative pairs to optimize learning
            hard_pairs = self.miner(embeddings, labels)

            # --- Metric Learning Loss ---
            # Computes triplet or margin-based loss using the mined pairs
            loss = self.loss_fn(embeddings, labels, hard_pairs)

            # --- Identity Loss ---
            if logits is not None:
                loss = loss + self.id_loss_weight * self.id_loss_fn(logits, labels)


            # --- Backpropagation with Accumulation ---
            # Divides loss by accumulation steps to average gradients correctly
            loss = loss / accum_steps
            self.scaler.scale(loss).backward()

            # Update weights only after specified accumulation steps
            if (i + 1) % accum_steps == 0:
                # Gradients must be unscaled before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
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


    def _get_features(self, loader, name):
        """Extract Embeddings from Dataloader"""
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