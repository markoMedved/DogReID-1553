import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip

class DOGVideoREIDDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split_file,
            split="train",
            clip_len=16,
            transform=None,
            use_videos=True,
            world="closed"
    ):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world

        df = pd.read_csv(split_file)

        WOLRD = "SPLIT_OPEN_SET"
        if world == "closed":
            WORLD = "SPLIT_CLOSED_SET"
        
        df = pd.read_csv(split_file)

        if split == "train":
            df = df[df["SPLIT_CLOSED_SET"] == "train"]

        elif split == "query":
            df = df[df["SPLIT_CLOSED_SET"] == "query"]

        elif split == "gallery":
            df = df[df["SPLIT_CLOSED_SET"] == "gallery"]

        # remove identities with only one sample
        counts = df["DOG_ID"].value_counts()
        valid_ids = counts[counts > 1].index
        df = df[df["DOG_ID"].isin(valid_ids)]

        self.df = df.reset_index(drop=True)

        dog_ids = sorted(df["DOG_ID"].unique())
        self.id_map = {dog_id: i for i, dog_id in enumerate(dog_ids)}

    def __len__(self):
        return len(self.df)
    
    def _get_path(self, dog_id, video_id):

        folder = "Videos" if self.use_videos else "Images"

        return os.path.join(
            self.root_dir,
            folder,
            dog_id,
            video_id
        )
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        dog_id = row["DOG_ID"]
        video_id  = row["VIDEO_ID"]

        path = self._get_path(dog_id, video_id)

        clip = load_video_clip(path, self.clip_len)

        if self.transform:
            clip = self.transform(clip)

        label = self.id_map[dog_id]

        return clip, label, dog_id, video_id