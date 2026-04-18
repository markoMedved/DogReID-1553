import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip
import numpy as np
import random
from PIL import Image

class DOGVideoREIDDataset(Dataset):
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", label_map=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world
        self.split = split

        # Load and filter by split
        df = pd.read_csv(split_file)
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # Eval sets (query/gallery) often only have 1 sample per dog/clip.
        if self.split == "train":
            counts = df["DOG_ID"].value_counts()
            valid_ids = counts[counts > 1].index
            df = df[df["DOG_ID"].isin(valid_ids)]
        
        self.df = df.reset_index(drop=True)

        # Build or use provided mapping
        if label_map is None:
            dog_ids = sorted(self.df["DOG_ID"].unique())
            self.id_map = {dog_id: i for i, dog_id in enumerate(dog_ids)}
        else:
            self.id_map = label_map


        self._labels = self.df["DOG_ID"].map(lambda x: self.id_map.get(x, -1)).tolist()

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        """Used by MPerClassSampler"""
        return self._labels

    def _get_path(self, dog_id, video_id):
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        
        # Structure: Videos/dog_id/dog_id-video_id.mp4
        filename = f"{dog_id}-{video_id}.{ext}"
        return os.path.join(self.root_dir, folder, dog_id, filename)


    def __getitem__(self, idx):
            row = self.df.iloc[idx]
            dog_id = row["DOG_ID"]
            video_id = row["VIDEO_ID"]

            path = self._get_path(dog_id, video_id)
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Clip not found: {path}")

            # 1. Temporal Sampling
            # Returns (T, H, W, C) numpy array from decord
            clip = load_video_clip(
                path, 
                self.clip_len, 
                is_training=(self.split == "train")
            )

            # 2. Spatial Augmentation & Tensor Conversion
            if self.transform:
                transformed_frames = []
                
                if self.split == "train":
                    # Generate a single seed for the entire video clip
                    seed = np.random.randint(2147483647)
                    
                    for frame in clip:
                        # Sync seeds so every frame gets the EXACT same crop/flip
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        
                        # Bridge: NumPy (H,W,C) -> PIL Image -> Transform -> Tensor
                        pil_img = Image.fromarray(frame)
                        transformed_frames.append(self.transform(pil_img))
                else:
                    # During Evaluation: No randomness, just deterministic Resize/Normalize
                    for frame in clip:
                        pil_img = Image.fromarray(frame)
                        transformed_frames.append(self.transform(pil_img))
                
                # Stack frames into a single video tensor: (T, C, H, W)
                clip = torch.stack(transformed_frames)
            else:
                # Fallback: Basic conversion if no transform provided
                # Converts (T, H, W, C) -> (T, C, H, W) and scales to [0, 1]
                clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0

            # 3. Labeling
            # label is the mapped integer (0...N), dog_id is the original string UID
            label = self._labels[idx] 
            
            return clip, label, dog_id, video_id