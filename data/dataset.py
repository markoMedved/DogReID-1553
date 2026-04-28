import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from .video_utils import load_video_clip
import numpy as np
import random
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

COCO_DOG_CLASS = 16  # COCO class index for 'dog'
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YOLO_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "yolo11_dogs", "weights", "best.pt")

def load_yolo(custom_path: str = DEFAULT_YOLO_PATH, 
              fallback_model: str = "yolo11n.pt", 
              device: torch.device = None) -> tuple[YOLO, int]:
    """
    Returns (YOLO_Model_Object, Dog_Class_Index)
    """
    if os.path.exists(custom_path):
        print(f"-> Loading Custom YOLO: {custom_path}")
        model = YOLO(custom_path)
        class_id = 0  # Custom model dog index
    else:
        print(f"-> Custom weights not found. Fallback: {fallback_model}")
        model = YOLO(fallback_model)
        class_id = 16 # COCO dog index

    # Use CPU to avoid CUDA initialization errors in DataLoader workers
    target_device = device if device is not None else torch.device("cpu")
    model.to(target_device)
    return model, class_id

def detect_dog_box(yolo: YOLO, frame: Image.Image, dog_class_id: int, conf_threshold: float = 0.3):
    results = yolo(frame, verbose=False)[0]
    best_box = None
    best_conf = conf_threshold

    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        
        if cls == dog_class_id and conf > best_conf:
            best_conf = conf
            best_box = tuple(map(int, box.xyxy[0].tolist()))

    return best_box


def crop_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
    padding: float = 0.05,
) -> Image.Image:
    """
    Crop a PIL frame to the given box with optional padding.
    Falls back to the full frame if box is None.

    Args:
        frame:   PIL Image.
        box:     (x1, y1, x2, y2) or None.
        padding: Fractional padding added around the box (0.05 = 5%).
    Returns:
        Cropped (and padded) PIL Image, or original frame if no box.
    """
    if box is None:
        return frame  # fallback: full frame

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
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", 
                 label_map=None, custom_yolo_path="runs/detect/yolo11_dogs/weights/best.pt"):

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world
        self.split = split

        self.yolo_model, self.dog_class_id = load_yolo(custom_path=custom_yolo_path)

        # --- Load Split Data ---
        df = pd.read_csv(split_file)

        # --- Select Split Column Based on World Setting ---
        split_col = "SPLIT_CLOSED_SET" if world == "closed" else "SPLIT_OPEN_SET"
        df = df[df[split_col] == split]

        # --- Remove Identities with Only One Sample ---
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
        row      = self.df.iloc[idx]
        dog_id   = row["DOG_ID"]
        video_id = row["VIDEO_ID"]
        path     = self._get_path(dog_id, video_id)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        # --- Load raw frames ---
        if self.use_videos:
            clip = load_video_clip(path, self.clip_len, is_training=(self.split == "train"))
        else:
            img  = Image.open(path).convert("RGB")
            clip = [np.array(img)]

        # --- YOLO crop (per frame, on raw pixels, before transforms) ---
        if self.yolo_model is not None:
            cropped_clip = []
            for frame_arr in clip:
                pil_frame = Image.fromarray(frame_arr)
                box = detect_dog_box(self.yolo_model, pil_frame, self.dog_class_id) 
                pil_frame = crop_frame(pil_frame, box) 
                cropped_clip.append(np.array(pil_frame)) 
            clip = cropped_clip

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

            clip = torch.stack(transformed_frames)          # (T, C, 224, 224)
        else:
            clip = torch.from_numpy(
                np.array(clip)
            ).permute(0, 3, 1, 2).float() / 255.0

        return clip, self._labels[idx], dog_id, video_id