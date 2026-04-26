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

        # --- Load Split Data ---
        df = pd.read_csv(split_file)

        # --- Select Split Column Based on World Setting ---
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # --- Remove Identities with Only One Sample ---
        # This is strictly required for proper metric learning during training
        if self.split == "train":
            counts = df["DOG_ID"].value_counts()
            valid_ids = counts[counts > 1].index
            df = df[df["DOG_ID"].isin(valid_ids)]
        
        self.df = df.reset_index(drop=True)

        # --- Store Dog IDs for External Access ---
        # Accessed by the dataloader to facilitate sampling logic
        self.dog_ids = self.df["DOG_ID"].tolist()

        # --- Build Dog ID to Label Mapping ---
        if label_map is None:
            dog_ids = sorted(self.df["DOG_ID"].unique())
            self.id_map = {dog_id: i for i, dog_id in enumerate(dog_ids)}
        else:
            self.id_map = label_map

        # --- Assign Integer Labels for Training ---
        self._labels = self.df["DOG_ID"].map(
            lambda x: self.id_map.get(x, -1)
        ).tolist()

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        # Property accessed directly by MPerClassSampler
        return self._labels

    def _get_path(self, dog_id, video_id):

        # --- Choose Dataset Folder and Extension ---
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        
        # --- Construct Filename Based on Dataset Format ---
        filename = f"{dog_id}-{video_id}.{ext}"

        return os.path.join(self.root_dir, folder, dog_id, filename)


    def __getitem__(self, idx):
            row = self.df.iloc[idx]
            dog_id, video_id = row["DOG_ID"], row["VIDEO_ID"]
            path = self._get_path(dog_id, video_id)
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")

            # --- Data Loading ---
            if self.use_videos:
                clip = load_video_clip(path, self.clip_len, is_training=(self.split == "train"))
            else:
                img = Image.open(path).convert("RGB")
                clip = [np.array(img)]

            # --- Transformation Pipeline ---
            if self.transform:
                transformed_frames = []
                seed = np.random.randint(2147483647)

                for frame in clip:
                    if self.split == "train":
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                    pil_img = Image.fromarray(frame)
                    transformed_frames.append(self.transform(pil_img))

                clip = torch.stack(transformed_frames)
            else:
                # Convert tensor format: (T, H, W, C) -> (T, C, H, W)
                clip = torch.from_numpy(np.array(clip)).permute(0, 3, 1, 2).float() / 255.0

            return clip, self._labels[idx], dog_id, video_id