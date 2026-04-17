import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip
import numpy as np

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

        # CRITICAL FIX 1: Only filter for multiple samples in training.
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

        # Pre-calculate labels for the Sampler
        # We use .get(dog_id, -1) to ensure it doesn't crash on unknown IDs
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
            # In some datasets, IDs in CSV might have prefixes or different casing
            raise FileNotFoundError(f"Clip not found: {path}")

        # Load raw numpy frames
        clip = load_video_clip(path, self.clip_len)

        # Apply transformations (Numpy -> PIL -> Tensor)
        if self.transform:
            clip = self.transform(clip)

        # Final label check
        label = self.id_map.get(dog_id, -1) 
        
        return clip, label, dog_id, video_id