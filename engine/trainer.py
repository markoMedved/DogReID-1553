import os
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
import torch.nn as nn

class Trainer:
    "Class that has the training logic"
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner,
                 scheduler=None, oa_sched_dict=None):
        # --- Move Model to Compute Device ---
        self.model = model.to(cfg.device)

        # --- Dataloaders ---
        self.train_loader = train_loader
        self.query_loader = query_loader # Validation
        self.gallery_loader = gallery_loader # Validation

        # --- Training Configuration ---
        self.optimizer = optimizer
        self.scheduler = scheduler # Stepped once per epoch, may be None
        self.oa_sched_dict = oa_sched_dict
        self.total_iters = 0
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
        if amp == 'bf16' and self.device.type == 'cuda' and not torch.cuda.is_bf16_supported():
            print("[amp] CUDA device does not support hardware bf16 (e.g. V100); falling back to fp16")
            amp = 'fp16'
        self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(amp)
        self.amp_enabled = self.amp_dtype is not None and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=self.amp_enabled and self.amp_dtype is torch.float16
        )
        if self.amp_enabled:
            print(f"[amp] mixed precision enabled ({amp})")

        # --- History Tracking & Checkpointing ---
        self.history = {
            "epochs": [],
            "losses": [],
            "lrs": [],
            "sub_epochs": [],
            "cls_losses": [],
            "tri_losses": [],
            "eval_epochs": [],
            "val_losses": [],
            "val_pos_dists": [],
            "rank1": [],
            "rank5": [],
            "mAP": []
        }
        self.best_mAP = 0.0
        self.best_r1 = 0.0
        self.start_epoch = 0

        # --- Clean Output Dir if Starting From Scratch ---
        if not getattr(cfg, "resume", False) and self.cfg.output_dir.exists():
            for f in self.cfg.output_dir.iterdir():
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
            print(f"[trainer] Starting from scratch. Cleared prior files in {self.cfg.output_dir}")

        # --- Resume Checkpoint if Requested ---
        if getattr(cfg, "resume", False):
            latest_path = self.cfg.output_dir / "latest_model.pth"
            if not latest_path.exists():
                candidates = []
                leg_dir = getattr(self.cfg, "legacy_output_dir", None)
                if leg_dir:
                    candidates.append(leg_dir / "latest_model.pth")
                root = getattr(self.cfg, "project_root", None)
                if root:
                    if hasattr(self.cfg, "legacy_run_name"):
                        candidates.append(root / "checkpoints_o" / self.cfg.legacy_run_name / "latest_model.pth")
                    if hasattr(self.cfg, "run_name"):
                        candidates.append(root / "checkpoints_o" / self.cfg.run_name / "latest_model.pth")
                for cand in candidates:
                    if cand.exists():
                        latest_path = cand
                        print(f"[resume] Note: found checkpoint under fallback path: {latest_path}")
                        break
            if latest_path.exists():
                print(f"[resume] Loading checkpoint from {latest_path} ...")
                ckpt = torch.load(latest_path, map_location=self.device)
                if "state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["state_dict"])
                if "optimizer" in ckpt and self.optimizer is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                if "scheduler" in ckpt and ckpt.get("scheduler") is not None and self.scheduler is not None:
                    try:
                        self.scheduler.load_state_dict(ckpt["scheduler"])
                    except Exception as e:
                        print(f"[resume] Warning loading scheduler: {e}")
                if self.oa_sched_dict and ckpt.get("oa_sched"):
                    for k, sd in ckpt["oa_sched"].items():
                        if k in self.oa_sched_dict:
                            self.oa_sched_dict[k].load_state_dict(sd)
                    print(f"[resume] Restored OpenAnimals schedulers: {list(ckpt['oa_sched'].keys())}")
                self.total_iters = ckpt.get("total_iters", 0)
                if ckpt.get("scaler") is not None and self.scaler.is_enabled():
                    self.scaler.load_state_dict(ckpt["scaler"])
                self.start_epoch = ckpt.get("epoch", 0)
                print(f"[resume] total_iters = {self.total_iters}, "
                      f"current LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                print(f"[resume] Successfully resumed model state from epoch {self.start_epoch}")

                hist_file = latest_path.parent / "training_history.json"
                if not hist_file.exists():
                    hist_file = self.cfg.output_dir / "training_history.json"
                if hist_file.exists():
                    try:
                        with open(hist_file, "r") as f:
                            self.history = json.load(f)
                        if self.history.get("mAP") and len(self.history["mAP"]) > 0:
                            self.best_mAP = max(self.history["mAP"]) / 100.0
                        if self.history.get("rank1") and len(self.history["rank1"]) > 0:
                            self.best_r1 = max(self.history["rank1"]) / 100.0
                        print(f"[resume] Loaded history ({len(self.history.get('epochs', []))} epochs). Best mAP so far: {self.best_mAP:.2%}")
                    except Exception as e:
                        print(f"[resume] Warning loading history: {e}")

                if latest_path.parent != self.cfg.output_dir:
                    src_best = latest_path.parent / "best_model.pth"
                    dst_best = self.cfg.output_dir / "best_model.pth"
                    if src_best.exists() and not dst_best.exists():
                        import shutil
                        try:
                            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src_best, dst_best)
                            print(f"[resume] Copied best model from {src_best} -> {dst_best}")
                        except Exception as e:
                            print(f"[resume] Warning copying best model: {e}")
            else:
                print(f"[resume] Checkpoint {latest_path} not found. Starting from scratch (epoch 0).")


    def _save_and_plot_history(self):
        try:
            self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
            hist_file = self.cfg.output_dir / "training_history.json"
            with open(hist_file, "w") as f:
                json.dump(self.history, f, indent=2)

            import sys
            root_dir = str(getattr(self.cfg, "project_root", Path(__file__).resolve().parent.parent))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)

            from evaluation.plot_training_logs import plot_metrics
            plot_file = self.cfg.output_dir / "training_curves.png"
            plot_metrics(self.history, str(plot_file), title=f"Training & Validation: {self.cfg.run_name}")
        except Exception as e:
            print(f"[history] Warning: failed to save/plot metrics: {e}")


    def train(self):
        """Train the model"""
        # --- Get the validation split ratio ---
        val_split = getattr(self.cfg, 'val_split', 0)

        # --- Main Training Loop ---
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.current_epoch = epoch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run one full training epoch
            avg_loss, avg_cls, avg_tri = self.train_epoch(epoch)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # One learning rate per parameter group: deduplicate unique LRs
            unique_lrs = []
            for g in self.optimizer.param_groups:
                lr_str = f"{g['lr']:.2e}"
                if lr_str not in unique_lrs:
                    unique_lrs.append(lr_str)
            lrs = " / ".join(unique_lrs)
            print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | LR: {lrs}")

            # Save history entry for this epoch
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["epochs"].append(epoch)
            self.history["losses"].append(float(avg_loss))
            self.history["lrs"].append(float(current_lr))
            if avg_cls is not None and avg_tri is not None:
                self.history["sub_epochs"].append(epoch)
                self.history["cls_losses"].append(float(avg_cls))
                self.history["tri_losses"].append(float(avg_tri))

            # --- Learning Rate Schedule ---
            # Stepped per epoch, matching the warmup and milestone units
            if self.oa_sched_dict and "lr_sched" in self.oa_sched_dict:
                oa_cfg = getattr(self.model, "oa_cfg", None)
                delay_epochs = oa_cfg.SOLVER.DELAY_EPOCHS if oa_cfg else 0
                warmup_iters = oa_cfg.SOLVER.WARMUP_ITERS if oa_cfg else 1000
                if self.total_iters >= warmup_iters and (epoch + 1) > delay_epochs:
                    self.oa_sched_dict["lr_sched"].step()
            elif self.scheduler is not None:
                self.scheduler.step()

            # --- Validation Evaluation (runs every epoch if eval_period=1) ---
            if self.cfg.world == "closed" and self.query_loader is not None:
                if (epoch + 1) % self.cfg.eval_period == 0 or (epoch + 1) == self.cfg.epochs:
                    r1, r5, mAP, val_loss, val_pos_dist = self.evaluate()
                    self.history["eval_epochs"].append(epoch + 1)
                    self.history["val_losses"].append(float(val_loss))
                    self.history["val_pos_dists"].append(float(val_pos_dist))
                    self.history["rank1"].append(float(r1 * 100))
                    self.history["rank5"].append(float(r5 * 100))
                    self.history["mAP"].append(float(mAP * 100))

                    # Track and save best model
                    if mAP > self.best_mAP:
                        self.best_mAP = mAP
                        self.best_r1 = r1
                        best_path = self.cfg.output_dir / "best_model.pth"
                        torch.save({
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'mAP': mAP,
                            'rank1': r1,
                            'val_loss': val_loss
                        }, best_path)
                        print(f"--> [BEST] New best model saved! (Rank-1: {r1:.2%}, mAP: {mAP:.2%}, Val Loss: {val_loss:.4f}) -> {best_path}")

            # Save latest checkpoint every epoch (overwritten each epoch; represents the final model at completion)
            latest_path = self.cfg.output_dir / "latest_model.pth"
            torch.save(self._training_state(epoch), latest_path)

            # Periodic checkpoint saving disabled (only latest_model.pth and best_model.pth are retained)
            save_period = getattr(self.cfg, 'save_period', 0)
            if save_period > 0 and (epoch + 1) % save_period == 0:
                checkpoint_path = self.cfg.output_dir / f"model_epoch_{epoch + 1}.pth"
                torch.save(self._training_state(epoch), checkpoint_path)
                print(f"--> Saved periodic checkpoint to {checkpoint_path}")

            # Update live training curves PNG and history JSON
            self._save_and_plot_history()        

        print(f"Training complete. Final model saved to: {self.cfg.output_dir / 'latest_model.pth'}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        # --- Put into train mode ---
        self.model.train()

        # Gradient accumulation helps simulate larger batch sizes
        accum_steps = getattr(self.cfg, 'accum_steps', 8) 

        running_loss = 0.0
        running_cls = 0.0
        running_tri = 0.0
        has_sub = False
        self.optimizer.zero_grad()

        # Initialize progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        # --- Batch Processing Loop ---
        for i, (videos, labels, dog_ids, video_ids) in enumerate(pbar):

            # Move data to the active device
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # OpenAnimals freeze training (SBS: backbone frozen for FREEZE_ITERS iterations,
            # with its BN layers in eval mode -- mirrors hooks.LayerFreeze + freeze optimizer)
            frozen_params = self._apply_layer_freeze()

            # Forward Pass & Loss Computation
            with torch.autocast(device_type=self.device.type,
                                dtype=self.amp_dtype,
                                enabled=self.amp_enabled):
                outputs = self.model(videos, targets=labels)

            if hasattr(self.model, "compute_loss"):
                total_loss, loss_triplet, loss_id = self.model.compute_loss(outputs, labels)
                has_sub = True
                running_cls += loss_id.item()
                running_tri += loss_triplet.item()
            else:
                if isinstance(outputs, tuple):
                    embeddings, logits = outputs
                else:
                    embeddings, logits = outputs, None

                # Losses are computed in float32; metric learning is sensitive to
                # reduced precision in the distance matrix
                if isinstance(embeddings, (list, tuple)):
                    embeddings = [e.float() for e in embeddings]
                else:
                    embeddings = embeddings.float()
                if logits is not None:
                    if isinstance(logits, (list, tuple)):
                        logits = [l.float() for l in logits]
                    else:
                        logits = logits.float()

                # --- Triplet Loss Computation ---
                if getattr(self.model, "is_openanimals", False):
                    from openanimals.modeling.losses.triplet_loss import triplet_loss as oa_triplet_loss
                    tri_cfg = self.model.oa_cfg.MODEL.LOSSES.TRI
                    feats = embeddings if isinstance(embeddings, (list, tuple)) else [embeddings]
                    # Raw (unnormalised) features; NORM_FEAT decides euclidean vs cosine as in
                    # OpenAnimals. Multiple feature sets (MGN: b1,b2,b3,b22,b33) are averaged,
                    # i.e. x0.2 each, matching MGN.losses.
                    loss_triplet = sum(
                        oa_triplet_loss(
                            f, labels,
                            margin=tri_cfg.MARGIN,
                            norm_feat=tri_cfg.NORM_FEAT,
                            hard_mining=tri_cfg.HARD_MINING
                        ) for f in feats
                    ) / len(feats) * tri_cfg.SCALE
                else:
                    hard_pairs = self.miner(embeddings, labels)
                    loss_triplet = self.loss_fn(embeddings, labels, hard_pairs)

                # --- Metric Learning & Identity Loss ---
                if logits is not None and getattr(self.cfg, 'num_classes', 0) > 0:
                    if isinstance(logits, (list, tuple)):
                        loss_id = sum(self.id_loss_fn(l, labels) for l in logits) / len(logits)
                    else:
                        loss_id = self.id_loss_fn(logits, labels)
                    total_loss = loss_triplet + self.id_loss_weight * loss_id
                    has_sub = True
                    running_cls += loss_id.item()
                    running_tri += loss_triplet.item()
                else:
                    total_loss = loss_triplet

            # --- Backpropagation with Accumulation ---
            # Divides loss by accumulation steps to average gradients correctly
            loss = total_loss / accum_steps
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights after specified accumulation steps or at the end of epoch
            is_last_step = (i + 1) == len(self.train_loader)
            if (i + 1) % accum_steps == 0 or is_last_step:
                # Frozen params get no update at all (grad=None also skips weight decay),
                # as in OpenAnimals' optimizer_wfl_step
                for p in frozen_params:
                    p.grad = None

                # OpenAnimals has gradient clipping disabled (SOLVER.CLIP_GRADIENTS.ENABLED=False)
                clip = not getattr(self.model, "is_openanimals", False)
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    if clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # OpenAnimals Warmup LR stepping per iteration
                self.total_iters += 1
                if self.oa_sched_dict and "warmup_sched" in self.oa_sched_dict:
                    oa_cfg = getattr(self.model, "oa_cfg", None)
                    warmup_iters = oa_cfg.SOLVER.WARMUP_ITERS if oa_cfg else 1000
                    if self.total_iters <= warmup_iters:
                        self.oa_sched_dict["warmup_sched"].step()

            # --- Update Progress Logging ---
            running_loss += total_loss.item()
            if has_sub:
                pbar.set_postfix_str(f"loss={total_loss.item():.4f} [tri:{loss_triplet.item():.3f} id:{loss_id.item():.3f}]")
            else:
                pbar.set_postfix(loss=total_loss.item())

        n_batches = len(self.train_loader)
        avg_loss = running_loss / n_batches
        avg_cls = (running_cls / n_batches) if has_sub else None
        avg_tri = (running_tri / n_batches) if has_sub else None
        return avg_loss, avg_cls, avg_tri

    def _training_state(self, epoch):
        """Everything needed to resume exactly: weights, optimizer, all schedulers,
        the iteration counter (drives warmup/freeze) and the AMP grad scaler."""
        return {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'oa_sched': {k: v.state_dict() for k, v in self.oa_sched_dict.items()} if self.oa_sched_dict else None,
            'total_iters': self.total_iters,
            'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }

    def _apply_layer_freeze(self):
        """Freeze cfg.MODEL.FREEZE_LAYERS for the first SOLVER.FREEZE_ITERS iterations.
        Returns the list of parameters whose gradients must be dropped this step."""
        oa_cfg = getattr(self.model, "oa_cfg", None)
        if oa_cfg is None:
            return []
        layers = list(oa_cfg.MODEL.FREEZE_LAYERS)
        freeze_iters = oa_cfg.SOLVER.FREEZE_ITERS
        if not layers or freeze_iters <= 0:
            return []

        modules = [m for n, m in self.model.oa_model.named_children() if n in layers]
        if self.total_iters < freeze_iters:
            if not getattr(self, "_layers_frozen", False):
                print(f"[freeze] Freezing {layers} for the first {freeze_iters} iterations")
                self._layers_frozen = True
            for m in modules:
                m.eval()  # BN in frozen layers uses running stats (re-applied every step
                          # because train_epoch() calls model.train())
            return [p for m in modules for p in m.parameters()]

        if getattr(self, "_layers_frozen", False):
            print(f"[freeze] Unfreezing {layers} at iteration {self.total_iters}")
            for m in modules:
                m.train()
            self._layers_frozen = False
        return []

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
            dist_mat = 1.0 - torch.mm(q_f, g_f.t())  # Calculate cosine distance
            dist_np = dist_mat.numpy()
            q_np = q_pids.numpy()
            g_np = g_pids.numpy()

            # Compute validation matching distances
            match_mask = (q_np[:, None] == g_np[None, :])
            val_pos_dist = float(dist_np[match_mask].mean()) if np.any(match_mask) else 0.0
            val_neg_dist = float(dist_np[~match_mask].mean()) if np.any(~match_mask) else 0.0

            # Compute Validation Triplet Ranking Loss (batch-hard retrieval loss)
            valid_queries = match_mask.any(axis=1)
            if np.any(valid_queries):
                pos_dists = np.where(match_mask, dist_np, -np.inf)
                hard_pos = np.max(pos_dists, axis=1)

                neg_dists = np.where(~match_mask, dist_np, np.inf)
                hard_neg = np.min(neg_dists, axis=1)

                margin = getattr(self.cfg, 'margin', 0.3)
                triplet_losses = np.maximum(0.0, hard_pos - hard_neg + margin)
                val_loss = float(np.mean(triplet_losses[valid_queries]))
            else:
                val_loss = 0.0

            r1, r5, mAP = self.calculate_cmc_map(dist_np, q_np, g_np)
            
            print(f"Eval (Closed) -> Val Loss: {val_loss:.4f} | Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, mAP: {mAP:.2%} | PosDist: {val_pos_dist:.4f}, NegDist: {val_neg_dist:.4f}")
            return r1, r5, mAP, val_loss, val_pos_dist

        return 0.0, 0.0, 0.0, 0.0, 0.0


    def _get_features(self, loader, name):
        """Extract Embeddings from Dataloader"""
        feats, pids = [], []

        with torch.autocast(device_type=self.device.type,
                            dtype=self.amp_dtype,
                            enabled=self.amp_enabled):
            for batch in tqdm(loader, desc=name):
                clips = batch[0].to(self.device)
                labels = batch[1]

                # Forward pass to get features
                f = self.model(clips)

                if isinstance(f, tuple):
                    f = f[0]

                # Normalize embeddings -> Allows cosine similarity via dot product
                f = F.normalize(f, p=2, dim=1)

                feats.append(f.cpu())
                pids.extend(labels.tolist())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        # Everything needed to reproduce the run and to fill in the settings
        # column of a results table
        allowed_keys = ['lr', 'margin', 'weight_decay', 'batch_size',
                        'k', 'num_ids', 'model', 'world', 'clip_len', 'epochs',
                        "accum_steps", "num_workers", "chunk_size",
                        # Re-ID method and architecture
                        "backbone", "reid_method", "pooling_type", "img_size",
                        "num_classes", "dinov2_variant", "megadescriptor_variant",
                        "jpm_parts", "jpm_shift", "jpm_shuffle_groups",
                        # Optimization
                        "full_finetune", "unfreeze_blocks", "id_loss_weight",
                        "warmup_epochs", "warmup_factor", "lr_milestones",
                        "lr_gamma", "amp",
                        # Augmentation
                        "aug_pad", "re_prob", "run_name"]

        params_to_save = {}

        for key in allowed_keys:

            if hasattr(self.cfg, key):
                params_to_save[key] = getattr(self.cfg, key)

            elif isinstance(self.cfg, dict) and key in self.cfg:
                params_to_save[key] = self.cfg[key]

        with open(meta_path, 'w') as f:
            json.dump(params_to_save, f, indent=4)
            
        print(f"Saved weights and metadata to: {target_dir}")