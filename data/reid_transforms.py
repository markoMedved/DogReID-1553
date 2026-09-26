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
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(pad),
        transforms.RandomCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        RandomErasing(probability=getattr(cfg, "re_prob", 0.5)),
    ])


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
            out.append(self.frame_tf(frame))

        return torch.stack(out, dim=0)


def build_video_transforms(cfg, is_train: bool):
    """Build a clip-level transform that is consistent across frames."""
    return ClipTransform(build_transforms(cfg, is_train))
