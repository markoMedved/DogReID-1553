import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip


class DOGVideoREIDDataset(Dataset):
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", label_map=None):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world

        df = pd.read_csv(split_file)
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # Filter IDs with at least 2 samples (necessary for Triplet Loss)
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

        # Pre-calculate labels for the Sampler (much faster!)
        self._labels = self.df["DOG_ID"].map(self.id_map).tolist()

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self._labels

    def _get_path(self, dog_id, video_id):
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        return os.path.join(self.root_dir, folder, dog_id, f"{dog_id}-{video_id}.{ext}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dog_id = row["DOG_ID"]
        video_id = row["VIDEO_ID"]

        path = self._get_path(dog_id, video_id)
        clip = load_video_clip(path, self.clip_len) # clip: (T, C, H, W)

        if self.transform:
            clip = self.transform(clip)

        label = self.id_map[dog_id]
        return clip, label, dog_id, video_id