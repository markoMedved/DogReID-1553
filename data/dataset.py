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

        # load split CSV
        df = pd.read_csv(split_file)

    

        # choose correct split column depending on world setting
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # remove identities with only one sample (needed for metric learning)
        if self.split == "train":
            counts = df["DOG_ID"].value_counts()
            valid_ids = counts[counts > 1].index
            df = df[df["DOG_ID"].isin(valid_ids)]
        
        self.df = df.reset_index(drop=True)

        # store dog ids for external access (used by dataloader)
        self.dog_ids = self.df["DOG_ID"].tolist()

        # build dog_id → label mapping
        if label_map is None:
            dog_ids = sorted(self.df["DOG_ID"].unique())
            self.id_map = {dog_id: i for i, dog_id in enumerate(dog_ids)}
        else:
            self.id_map = label_map

        # integer labels used during training
        self._labels = self.df["DOG_ID"].map(
            lambda x: self.id_map.get(x, -1)
        ).tolist()

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        # used by MPerClassSampler
        return self._labels

    def _get_path(self, dog_id, video_id):

        # choose dataset folder
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        
        # dataset naming format
        filename = f"{dog_id}-{video_id}.{ext}"

        return os.path.join(self.root_dir, folder, dog_id, filename)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        dog_id = row["DOG_ID"]
        video_id = row["VIDEO_ID"]

        # resolve file path
        path = self._get_path(dog_id, video_id)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Clip not found: {path}")

        # temporal frame sampling (T,H,W,C)
        clip = load_video_clip(
            path,
            self.clip_len,
            is_training=(self.split == "train")
        )

        # apply spatial transforms frame-by-frame
        if self.transform:

            transformed_frames = []

            if self.split == "train":
                # same augmentation for every frame in the clip
                seed = np.random.randint(2147483647)

                for frame in clip:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    pil_img = Image.fromarray(frame)
                    transformed_frames.append(self.transform(pil_img))

            else:
                # deterministic transforms during evaluation
                for frame in clip:
                    pil_img = Image.fromarray(frame)
                    transformed_frames.append(self.transform(pil_img))

            # stack frames → (T,C,H,W)
            clip = torch.stack(transformed_frames)

        else:
            # fallback conversion (T,H,W,C) → (T,C,H,W)
            clip = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0

        # mapped integer label
        label = self._labels[idx]

        return clip, label, dog_id, video_id