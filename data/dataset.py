import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO

COCO_DOG_CLASS = 16  # COCO class index for 'dog' 


def load_yolo(model_name: str = "yolo11n.pt", device: torch.device = None) -> YOLO:
    """Load YOLOv11 model onto the specified device."""
    model = YOLO(model_name)
    if device is not None:
        model.to(device)
    return model


def _best_dog_box(result, conf_threshold: float = 0.3):
    """Pick the highest-confidence dog box from one YOLO result."""
    best_box = None
    best_conf = conf_threshold

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == COCO_DOG_CLASS and conf > best_conf:
            best_conf = conf
            best_box = tuple(map(int, box.xyxy[0].tolist()))

    return best_box


def detect_dog_boxes(
    yolo: YOLO,
    frames: list,
    conf_threshold: float = 0.3,
) -> list:
    """
    Run YOLO once over a whole clip and return one box per frame.

    Detection dominates data loading cost, so the frames of a clip are batched
    into a single call instead of one call per frame.
    """
    if not frames:
        return []

    results = yolo(frames, verbose=False)
    return [_best_dog_box(r, conf_threshold) for r in results]


def detect_dog_box(
    yolo: YOLO,
    frame: Image.Image,
    conf_threshold: float = 0.3,
) -> tuple[int, int, int, int] | None:
    """
    Run YOLO on a single PIL frame and return the highest-confidence dog box.
    """
    results = yolo(frame, verbose=False)[0]

    best_box  = None
    best_conf = conf_threshold  # only accept detections above this

    for box in results.boxes:
        cls  = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == COCO_DOG_CLASS and conf > best_conf:
            best_conf = conf
            best_box  = tuple(map(int, box.xyxy[0].tolist()))  

    return best_box  # None if nothing passed threshold


def crop_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
    padding: float = 0.05,
) -> Image.Image:
    """
    Crop a PIL frame to the given box with padding.
    Falls back to the full frame if box is None.
    """
    if box is None:
        return frame  # If no box passed the treshold: full frame

    W, H = frame.size
    x1, y1, x2, y3 = box

    # Add padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y3 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y3 = min(H, y3 + pad_y)

    return frame.crop((x1, y1, x2, y3))


class DOGVideoREIDDataset(Dataset):
    """Dataset for loading videos from our DogReID-1553 dataset"""
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", label_map=None, yolo_model: str | None = "yolo11n.pt"):
        
        # --- Load Config parameters ---
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world
        self.split = split

        # --- Build the detection model ---
        self.yolo = load_yolo(yolo_model, device=torch.device("cpu")) if yolo_model else None

        # --- Load Split Data ---
        df = pd.read_csv(split_file)

        # --- Select Split Column Based on World Setting ---
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # --- Remove Identities with Only One Sample For Training---
        if self.split == "train":
            counts = df["DOG_ID"].value_counts()
            valid_ids = counts[counts > 1].index
            df = df[df["DOG_ID"].isin(valid_ids)]
        
        self.df = df.reset_index(drop=True)

        # --- Store Dog IDs for External Access ---
        # Accessed by the dataloader for sampling logic
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
        # Needed for the MPerClassSampler
        return self._labels

    def _get_path(self, dog_id, video_id):

        # --- Choose Dataset Folder and Extension ---
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        
        # --- Construct Filename Based on Dataset Format ---
        filename = f"{dog_id}-{video_id}.{ext}"

        return os.path.join(self.root_dir, folder, dog_id, filename)


    def __getitem__(self, idx):
        # --- Build path for loading video ---
        row      = self.df.iloc[idx]
        dog_id   = row["DOG_ID"]
        video_id = row["VIDEO_ID"]
        path     = self._get_path(dog_id, video_id)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        # --- Load raw frames ---
        if self.use_videos:
            clip = load_video_clip(path, self.clip_len, is_training=(self.split == "train")) # If training the sampling is different 
        else:
            img  = Image.open(path).convert("RGB")
            clip = [np.array(img)]

        # --- YOLO detected bounging box crop ---
        # The whole clip is detected in one batched call; per-frame calls made
        # detection the dominant cost of data loading.
        if self.yolo is not None:
            pil_frames = [Image.fromarray(frame_arr) for frame_arr in clip]
            boxes = detect_dog_boxes(self.yolo, pil_frames)
            clip = [
                np.array(crop_frame(pil_frame, box))
                for pil_frame, box in zip(pil_frames, boxes)
            ]

        # --- Use the transforms ---
        if self.transform:
            # Convert all NumPy frames in the clip to PIL Images
            pil_clip = [Image.fromarray(frame) for frame in clip]
            
            # Pass the entire list of frames to reid_transforms at once
            transformed = self.transform(pil_clip)
            
            # Stack the list of frame tensors into a single clip tensor (T, C, H, W)
            if isinstance(transformed, list):
                clip = torch.stack(transformed)
            else:
                clip = transformed
        else:
            clip = torch.from_numpy(
                np.array(clip)
            ).permute(0, 3, 1, 2).float() / 255.0

        return clip, self._labels[idx], dog_id, video_id