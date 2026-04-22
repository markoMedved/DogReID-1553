import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import json
from ultralytics import YOLO

class Trainer:
    def __init__(self, model, train_loader, query_loader, gallery_loader, optimizer, cfg, loss_fn, miner):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.optimizer = optimizer
        self.device = cfg.device
        self.cfg = cfg
        
        self.loss_fn = loss_fn
        self.miner = miner

        # --- UPDATED: Load Fine-Tuned Specialist Model ---
        detector_path = 'runs/detect/dog_detector_closed_world/weights/best.pt'
        
        if os.path.exists(detector_path):
            print(f"🎯 Loading fine-tuned dog detector: {detector_path}")
            self.detector = YOLO(detector_path).to(self.device)
            # Custom training usually results in 'dog' being the first/only class (index 0)
            self.dog_class_id = 0 
        else:
            print("⚠️ Specialist weights not found at path! Falling back to yolov8n.pt")
            self.detector = YOLO('yolov8n.pt').to(self.device)
            self.dog_class_id = 16 # COCO Dog class

    def apply_detection_and_crop(self, videos):
        """
        Input: videos (B, C, T, H, W) 
        Output: Cropped and Resized videos (B, C, T, img_size, img_size)
        """
        B, C, T, H, W = videos.shape
        cropped_batch = []
        img_size = getattr(self.cfg, 'img_size', 224)

        for b in range(B):
            # 1. Extract middle frame carefully for the detector
            ref_frame = videos[b, :, T // 2, :, :]
            
            # Ensure 3-channel (RGB) shape (1, 3, H, W)
            if ref_frame.shape[0] != 3:
                ref_frame = ref_frame[0:1, :, :].repeat(3, 1, 1)
            
            input_tensor = ref_frame.unsqueeze(0)

            # 2. Normalize values for the detector (0.0 to 1.0)
            f_min, f_max = input_tensor.min(), input_tensor.max()
            if f_min < 0 or f_max > 1:
                input_tensor = (input_tensor - f_min) / (f_max - f_min + 1e-6)

            # 3. Run Specialist Detector
            with torch.no_grad():
                results = self.detector(input_tensor.to(self.device).float(), verbose=False, conf=0.25)[0]
            
            dog_boxes = [box for box in results.boxes if int(box.cls) == self.dog_class_id]
            
            if len(dog_boxes) > 0:
                # STRATEGY: Pick the LARGEST box (Area = w * h)
                # This ensures we crop the subject dog, not a tiny one in the background
                best_box = sorted(dog_boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]), reverse=True)[0]
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Clamp coordinates to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                # Crop original video sequence (C, T, H, W)
                crop = videos[b, :, :, y1:y2, x1:x2]
            else:
                # FALLBACK: No dog found. 
                # Instead of the full image, take a 75% center crop to reduce background bias
                h_margin, w_margin = int(H * 0.125), int(W * 0.125)
                crop = videos[b, :, :, h_margin:H-h_margin, w_margin:W-w_margin]

            # 4. Resize back to fixed input size
            # interpolate wants (N, C, H, W). We treat T as the Batch dimension.
            t_as_b = crop.permute(1, 0, 2, 3) 
            resized_crop = F.interpolate(t_as_b, size=(img_size, img_size), 
                                        mode='bilinear', align_corners=False)
            
            # Back to (C, T, H, W)
            cropped_batch.append(resized_crop.permute(1, 0, 2, 3))

        return torch.stack(cropped_batch)

    def train_epoch(self, epoch):
        self.model.train()
        accum_steps = getattr(self.cfg, 'accum_steps', 8) 
        running_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for i, (videos, labels, dog_ids, video_ids) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)

            # Apply specialist cropping before Re-ID forward pass
            # videos = self.apply_detection_and_crop(videos)

            # 1. Forward pass
            embeddings = self.model(videos)
            
            # 2. Mining & Loss
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

    def train(self):
        best_rank1 = 0.0
        val_split = getattr(self.cfg, 'val_split', 0)

        for epoch in range(self.cfg.epochs):
            self.current_epoch = epoch
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

            if val_split > 0:
                if (epoch + 1) % self.cfg.eval_period == 0:
                    rank1, rank5, mAP = self.evaluate()
                    
                    if rank1 > best_rank1:
                        best_rank1 = rank1
                        self.save_checkpoint("best_model.pth")
                
            self.save_checkpoint("last_model.pth")

        if val_split <= 0.01:
            print("!!! Final training run detected (val_split=0). Saving final model...")
            self.save_checkpoint("final_model.pth")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        
        q_f, q_pids = self._get_features(self.query_loader, "Querying")
        g_f, g_pids = self._get_features(self.gallery_loader, "Gallerying")

        if self.cfg.world == "closed":
            dist_mat = 1 - torch.mm(q_f, g_f.t())
            r1, r5, mAP = self.calculate_cmc_map(dist_mat.numpy(), q_pids.numpy(), g_pids.numpy())
            print(f"Eval (Closed) -> Rank-1: {r1:.2%}, Rank-5: {r5:.2%}, mAP: {mAP:.2%}")
            return r1, r5, mAP
        else:
            thresh, dir_curve, far_curve = self.dir_vs_far(q_f, q_pids, g_f, g_pids)
            idx_1pct = np.argmin(np.abs(far_curve - 0.01))
            dir_1 = dir_curve[idx_1pct]
            print(f"Eval (Open) -> DIR@1%FAR: {dir_1:.2%}")
            return dir_1, 0.0, 0.0

    def _get_features(self, loader, name):
        feats, pids = [], []
        for batch in tqdm(loader, desc=name):
            clips = batch[0].to(self.device)
            labels = batch[1]
            
            # Apply same cropping logic to evaluation data
            clips = self.apply_detection_and_crop(clips)
            
            f = self.model(clips)
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
            dir_list.append(((max_vals_known > thresh) & top_is_correct).float().mean().item() if known_sims.numel() > 0 else 0.0)
            far_list.append((max_vals_unknown > thresh).float().mean().item() if unknown_sims.numel() > 0 else 0.0)

        return thresholds.cpu().numpy(), np.array(dir_list), np.array(far_list)

    def save_checkpoint(self, filename):
        val_split = getattr(self.cfg, 'val_split', 0)
        target_dir = self.cfg.output_dir
        if val_split == 0:
            target_dir = os.path.join(target_dir, f"final_model_{self.cfg.model}")
            
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, filename)
        meta_path = path.replace(".pth", "_params.json")

        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        checkpoint_data = {
            'model': state_dict,
            'epoch': getattr(self, 'current_epoch', 'unknown'),
            'val_split': val_split
        }
        torch.save(checkpoint_data, path)

        allowed_keys = ['lr', 'margin', 'weight_decay', 'batch_size', 'k', 'model', 'world', 'clip_len']
        params_to_save = {key: getattr(self.cfg, key) for key in allowed_keys if hasattr(self.cfg, key)}

        with open(meta_path, 'w') as f:
            json.dump(params_to_save, f, indent=4)
            
        print(f"✅ Saved weights and metadata to: {target_dir}")