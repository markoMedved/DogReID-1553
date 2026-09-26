import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from ultralytics import YOLO

from .video_utils import load_video_clip

COCO_DOG_CLASS = 16  # COCO class index for 'dog'


def load_yolo(model_name: str = "yolo11n.pt", device: torch.device = None) -> YOLO:
    """Load YOLO model onto the specified device."""
    model = YOLO(model_name)
    if device is not None:
        model.to(device)
    return model


def detect_dog_box(
    yolo: YOLO,
    frame: Image.Image,
    conf_threshold: float = 0.3,
) -> tuple[int, int, int, int] | None:
    """Run YOLO on a single PIL frame and return the highest-confidence dog box."""
    results = yolo(frame, verbose=False)[0]

    best_box = None
    best_conf = conf_threshold

    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == COCO_DOG_CLASS and conf > best_conf:
            best_conf = conf
            best_box = tuple(map(int, box.xyxy[0].tolist()))  # (x1, y1, x2, y2)

    return best_box


def crop_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
    padding: float = 0.05,
) -> Image.Image:
    """Crop a PIL frame to the given box with optional padding."""
    if box is None:
        return frame  # fallback: full frame

    W, H = frame.size
    x1, y1, x2, y3 = box

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y3 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(W, x2 + pad_x)
    y3 = min(H, y3 + pad_y)

    return frame.crop((x1, y1, x2, y3))


def mask_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Mask out the dog bounding box by painting it black."""
    if box is None:
        return frame

    frame_copy = frame.copy()
    draw = ImageDraw.Draw(frame_copy)
    draw.rectangle(box, fill="black")
    return frame_copy


class DOGVideoREIDDataset(Dataset):
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", label_map=None,
                 mask_dog=False, force_yolo=True, yolo_model: str | None = "yolo11n.pt", bbox_file: str | None = None):

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world
        self.split = split
        self.mask_dog = mask_dog
        self.force_yolo = force_yolo

        # Load YOLO model for videos or fallback
        self.yolo = load_yolo(yolo_model, device=torch.device("cpu")) if yolo_model else None

        # --- Load Ground Truth Bounding Boxes (for images) ---
        self.gt_bboxes = {}
        if bbox_file is not None:
            if not os.path.exists(bbox_file):
                # Crash immediately if the file is missing
                raise FileNotFoundError(f"CRITICAL: bbox_file was requested but not found at: {bbox_file}")
            
            try:
                bbox_df = pd.read_csv(bbox_file)
                for _, row in bbox_df.iterrows():
                    # Convert (x_top_left, y_top_left, width, height) -> (x1, y1, x2, y2)
                    x1 = int(row["x_top_left"])
                    y1 = int(row["y_top_left"])
                    x2 = x1 + int(row["width"])
                    y2 = y1 + int(row["height"])
                    
                    
                    self.gt_bboxes[(str(row["DOG_ID"]), str(row["VIDEO_ID"]))] = (x1, y1, x2, y2)
                
                # Print a confirmation to your SLURM logs
                print(f"-> [SUCCESS] Loaded {len(self.gt_bboxes)} ground truth boxes from {bbox_file}")
                
            except Exception as e:
                # Catch pandas reading errors or missing column errors
                raise RuntimeError(f"CRITICAL: Failed to parse bbox_file. Are the columns correct? Error: {e}")

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
        return self._labels

    def _get_path(self, dog_id, video_id):
        folder = "Videos" if self.use_videos else "Images"
        ext = "mp4" if self.use_videos else "jpg"
        filename = f"{dog_id}-{video_id}.{ext}"
        return os.path.join(self.root_dir, folder, dog_id, filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dog_id, video_id = str(row["DOG_ID"]), str(row["VIDEO_ID"])
        path = self._get_path(dog_id, video_id)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        # --- Data Loading ---
        if self.use_videos:
            clip = load_video_clip(path, self.clip_len, is_training=(self.split == "train"))
        else:
            img = Image.open(path).convert("RGB")
            clip = [np.array(img)]

        # --- Bounding Box Resolution ---
        processed_clip = []
        for frame_arr in clip:
            pil_frame = Image.fromarray(frame_arr)

            # Check GT first if using images
            box = None
            if not self.use_videos and not self.force_yolo:
                box = self.gt_bboxes.get((dog_id, video_id))

            # Fall back to YOLO if force_yolo is True, or if GT box was missing
            if box is None and self.yolo is not None:
                box = detect_dog_box(self.yolo, pil_frame)

            # Apply masking or cropping
            if self.mask_dog:
                pil_frame = mask_frame(pil_frame, box)
            else:
                pil_frame = crop_frame(pil_frame, box)

            processed_clip.append(np.array(pil_frame))
        
        clip = processed_clip

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
            clip = torch.from_numpy(np.array(clip)).permute(0, 3, 1, 2).float() / 255.0

        return clip, self._labels[idx], dog_id, video_id