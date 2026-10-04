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

    # Normalize coordinates
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y3), max(y1, y3)

    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x1 = max(0, min(W, x_min - pad_x))
    y1 = max(0, min(H, y_min - pad_y))
    x2 = max(0, min(W, x_max + pad_x))
    y3 = max(0, min(H, y_max + pad_y))

    # If box is degenerate or empty, return original frame
    if x2 <= x1 or y3 <= y1:
        return frame

    return frame.crop((x1, y1, x2, y3))


def mask_frame(
    frame: Image.Image,
    box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Mask out the dog bounding box by painting it black."""
    if box is None:
        return frame

    W, H = frame.size
    x1, y1, x2, y3 = box
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y3), max(y1, y3)

    x1 = max(0, min(W, x_min))
    y1 = max(0, min(H, y_min))
    x2 = max(0, min(W, x_max))
    y3 = max(0, min(H, y_max))

    if x2 <= x1 or y3 <= y1:
        return frame

    frame_copy = frame.copy()
    draw = ImageDraw.Draw(frame_copy)
    draw.rectangle((x1, y1, x2, y3), fill="black")
    return frame_copy


class DOGVideoREIDDataset(Dataset):
    def __init__(self, root_dir, split_file, split="train", clip_len=16, 
                 transform=None, use_videos=True, world="closed", label_map=None,
                 mask_dog=False, force_yolo=False, yolo_model: str | None = "yolo11n.pt", bbox_file: str | None = None):

        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.use_videos = use_videos
        self.world = world
        self.split = split
        self.mask_dog = mask_dog
        self.force_yolo = force_yolo

        # Load YOLO model only if explicitly requested
        self.yolo = load_yolo(yolo_model, device=torch.device("cpu")) if (self.force_yolo and yolo_model) else None

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

        # --- Bounding Box Resolution & Crop/Mask ---
        pil_frames = [Image.fromarray(f) for f in clip]
        boxes = [None] * len(pil_frames)

        # Check GT first if not force_yolo
        if not self.force_yolo:
            gt_box = self.gt_bboxes.get((dog_id, video_id))
            if gt_box is not None:
                boxes = [gt_box] * len(pil_frames)

        # Use YOLO if boxes are not already found and yolo is available
        if any(b is None for b in boxes) and self.yolo is not None:
            if not self.use_videos:
                boxes = [detect_dog_box(self.yolo, pil_frames[0])]
            else:
                boxes = detect_dog_boxes(self.yolo, pil_frames)

        processed_clip = []
        for pil_frame, box in zip(pil_frames, boxes):
            if self.mask_dog:
                pil_frame = mask_frame(pil_frame, box)
            else:
                pil_frame = crop_frame(pil_frame, box)
            processed_clip.append(np.array(pil_frame))

        clip = processed_clip

        # --- Transformation Pipeline ---
        if self.transform:
            pil_clip = [Image.fromarray(frame) for frame in clip]
            if hasattr(self.transform, "frame_tf"):
                clip = self.transform(pil_clip)
            else:
                transformed_frames = []
                seed = np.random.randint(2147483647)
                for pil_img in pil_clip:
                    if self.split == "train":
                        random.seed(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                    transformed_frames.append(self.transform(pil_img))
                clip = torch.stack(transformed_frames)
        else:
            clip = torch.from_numpy(np.array(clip)).permute(0, 3, 1, 2).float() / 255.0

        return clip, self._labels[idx], dog_id, video_id