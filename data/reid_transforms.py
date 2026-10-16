"""Re-ID augmentation pipeline.

Follows the augmentation policy of Luo et al., "Bag of Tricks and a Strong
Baseline for Deep Person Re-Identification" (CVPRW 2019 / IEEE TMM 2020):
resize, horizontal flip, pad and random crop, normalize, random erasing.

Random erasing follows Zhong et al., "Random Erasing Data Augmentation"
(AAAI 2020). arXiv:1708.04896

Input size defaults to a non-square 256x192. The dog bounding boxes in
bounding_boxes.csv have a geometric-mean width:height ratio of 0.81, so a
non-square input matches the data more closely than a square resize. Use
252x182 for DINOv2 so both dimensions are multiples of its patch size.
"""

import math
import random

import torch
import numpy as np
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomErasing:
    """Randomly erase a rectangular region of a normalized image tensor."""

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.0, 0.0, 0.0)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img

        C, H, W = img.shape
        area = H * W

        for _ in range(100):
            target_area = random.uniform(self.sl, self.sh) * area
            aspect = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect)))
            w = int(round(math.sqrt(target_area / aspect)))

            if h < H and w < W:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)
                for c in range(C):
                    img[c, x1:x1 + h, y1:y1 + w] = self.mean[c] if c < len(self.mean) else 0.0
                return img

        return img


def build_transforms(cfg, is_train: bool):
    """Build the per-frame transform pipeline."""
    size = tuple(getattr(cfg, "img_size", (256, 192)))

    if not is_train:
        # --- Deterministic Evaluation Transforms ---
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    # --- Training Augmentations ---
    pad = getattr(cfg, "aug_pad", 10)
    tfs = []
    autoaug_prob = getattr(cfg, "autoaug_prob", 0.0)
    if autoaug_prob > 0:
        # OpenAnimals applies AutoAugment first, before resizing (SBS: p=0.1)
        import sys
        from pathlib import Path
        oa_dir = str(Path(__file__).resolve().parent.parent / "OpenAnimals")
        if oa_dir not in sys.path:
            sys.path.insert(0, oa_dir)
        from openanimals.data.transforms.autoaugment import AutoAugment
        tfs.append(transforms.RandomApply([AutoAugment()], p=autoaug_prob))
    tfs += [
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(pad),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        RandomErasing(probability=getattr(cfg, "re_prob", 0.5)),
    ]
    return transforms.Compose(tfs)


class ClipTransform:
    """Apply one transform pipeline consistently across all frames of a clip.

    Sampling the augmentation independently per frame injects variation that
    the temporal pooling then has to compensate for, so the random state is
    fixed once per clip.
    """

    def __init__(self, frame_tf):
        self.frame_tf = frame_tf

    def __call__(self, frames):
        seed = torch.randint(0, 2 ** 31 - 1, (1,)).item()

        out = []
        for frame in frames:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            out.append(self.frame_tf(frame))

        return torch.stack(out, dim=0)


class TFCLIPSeqTransform:
    """Exact port of TF-CLIP's SeqT augmentation pipeline (utils/seqtransforms.py).

    Training:
      1. RectScale(height, width, Image.BILINEAR) (no Pad, no RandomCrop)
      2. Sequence-level RandomHorizontalFlip(p=0.5)
      3. Per-frame RandomSizedEarser(sl=0.02, sh=0.2, asratio=0.3, p=re_prob)
      4. ToTensor() + Normalize(IMAGENET_MEAN, IMAGENET_STD)
    Evaluation / Stage 1:
      1. RectScale(height, width, Image.BILINEAR)
      2. ToTensor() + Normalize(IMAGENET_MEAN, IMAGENET_STD)
    """

    def __init__(self, size=(224, 224), is_train=True, re_prob=0.5, sl=0.02, sh=0.2, asratio=0.3):
        from PIL import Image
        self.height, self.width = int(size[0]), int(size[1])
        self.is_train = is_train
        self.re_prob = re_prob
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.interpolation = Image.BILINEAR
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        # Marker attribute so DOGVideoREIDDataset passes the full clip list directly
        self.frame_tf = True

    def _erase_frame(self, frame):
        from PIL import Image
        if random.uniform(0.0, 1.0) > self.re_prob:
            return frame
        W, H = frame.size
        area = H * W
        for _ in range(100):
            Se = random.uniform(self.sl, self.sh) * area
            re = random.uniform(self.asratio, 1.0 / self.asratio)
            He = np.sqrt(Se * re)
            We = np.sqrt(Se / re)
            xe = random.uniform(0, W - We)
            ye = random.uniform(0, H - He)
            if xe + We <= W and ye + He <= H and xe > 0 and ye > 0:
                x1 = int(np.ceil(xe))
                y1 = int(np.ceil(ye))
                x2 = int(np.floor(x1 + We))
                y2 = int(np.floor(y1 + He))
                if x2 > x1 and y2 > y1:
                    frame = frame.copy()
                    Rc = random.randint(0, 255)
                    Gc = random.randint(0, 255)
                    Bc = random.randint(0, 255)
                    patch = Image.new("RGB", (x2 - x1, y2 - y1), (Rc, Gc, Bc))
                    frame.paste(patch, (x1, y1))
                    break
        return frame

    def __call__(self, frames):
        from PIL import Image
        # 1. RectScale (Bilinear resize without padding or random crop)
        scaled = [
            f if f.size == (self.width, self.height) else f.resize((self.width, self.height), self.interpolation)
            for f in frames
        ]

        if self.is_train:
            # 2. Sequence-level RandomHorizontalFlip (p=0.5 across whole clip)
            if random.random() < 0.5:
                scaled = [f.transpose(Image.FLIP_LEFT_RIGHT) for f in scaled]
            # 3. Per-frame RandomSizedEarser (sl=0.02, sh=0.2, asratio=0.3, p=0.5)
            scaled = [self._erase_frame(f) for f in scaled]

        # 4. ToTensor + Normalize
        out = [self.normalize(self.to_tensor(f)) for f in scaled]
        return torch.stack(out, dim=0)


def build_video_transforms(cfg, is_train: bool):
    """Build a clip-level transform that is consistent across frames."""
    bb = str(getattr(cfg, "backbone", getattr(cfg, "model", ""))).lower()
    if "tfclip" in bb or "tf_clip" in bb or bb == "clip":
        size = tuple(getattr(cfg, "img_size", (224, 224)))
        re_prob = getattr(cfg, "re_prob", 0.5)
        return TFCLIPSeqTransform(size=size, is_train=is_train, re_prob=re_prob)
    return ClipTransform(build_transforms(cfg, is_train))
