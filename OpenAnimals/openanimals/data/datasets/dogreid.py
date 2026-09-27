# encoding: utf-8
"""
DogReID-1553 Dataset registration for OpenAnimals.
Supports closed-set and open-set splits, with bbox cropping or full-image mode.
"""

import os
import os.path as osp
import logging
import numpy as np
import pandas as pd

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY

logger = logging.getLogger(__name__)


def _find_dogreid_root(root):
    """Automatically find the DogReID-1553 root directory containing splits.csv and Images/."""
    # 1. Environment variable
    env_root = os.getenv("DOGREID_DATASET_ROOT", "")
    if env_root and osp.exists(osp.join(env_root, "splits.csv")):
        return osp.abspath(env_root)

    # 2. Directly specified root
    if root and osp.exists(osp.join(root, "splits.csv")):
        return osp.abspath(root)
    if root and osp.exists(osp.join(root, "DogReID-1553", "splits.csv")):
        return osp.abspath(osp.join(root, "DogReID-1553"))

    # 3. Relative to this file: OpenAnimals/openanimals/data/datasets/../../../../
    candidate = osp.abspath(osp.join(osp.dirname(__file__), "../../../.."))
    if osp.exists(osp.join(candidate, "splits.csv")):
        return candidate

    # 4. Current working directory or parent
    cwd = os.getcwd()
    if osp.exists(osp.join(cwd, "splits.csv")):
        return osp.abspath(cwd)
    parent = osp.abspath(osp.join(cwd, ".."))
    if osp.exists(osp.join(parent, "splits.csv")):
        return parent

    raise FileNotFoundError(
        f"Could not locate DogReID-1553 dataset root directory containing splits.csv. "
        f"Searched: '{root}', '{candidate}', '{cwd}'. Please set DOGREID_DATASET_ROOT env var."
    )


@DATASET_REGISTRY.register()
class DogReID(ImageDataset):
    """DogReID-1553 Dataset (Closed-set split with bounding box cropping by default).

    DogReID-1553 contains 7,463 frames from 1,553 distinct dog identities.
    Closed-set: 776 train identities (3,788 images),
                777 test identities (1,677 query, 1,998 gallery).
    """
    dataset_name = "DogReID"

    def __init__(self, root='datasets', split_type='closed', img_load='bbox', **kwargs):
        self.root = root
        self.split_type = split_type.lower()
        assert self.split_type in ['closed', 'open'], f"split_type must be 'closed' or 'open', got {split_type}"
        assert img_load in ['bbox', 'full'], f"img_load must be 'bbox' or 'full', got {img_load}"
        self.img_load = img_load

        self.dataset_dir = _find_dogreid_root(self.root)
        self.splits_file = osp.join(self.dataset_dir, "splits.csv")
        self.bbox_file = osp.join(self.dataset_dir, "bounding_boxes.csv")
        self.images_dir = osp.join(self.dataset_dir, "Images")

        self.check_before_run([self.splits_file, self.images_dir])
        if self.img_load == 'bbox':
            self.check_before_run([self.bbox_file])

        splits_df = pd.read_csv(self.splits_file)
        split_col = 'SPLIT_CLOSED_SET' if self.split_type == 'closed' else 'SPLIT_OPEN_SET'

        bboxes_dict = {}
        if self.img_load == 'bbox' and osp.exists(self.bbox_file):
            bbox_df = pd.read_csv(self.bbox_file)
            for _, row in bbox_df.iterrows():
                x = max(0, int(row['x_top_left']))
                y = max(0, int(row['y_top_left']))
                w = int(row['width'])
                h = int(row['height'])
                bboxes_dict[(str(row['DOG_ID']), str(row['VIDEO_ID']))] = np.array([x, y, w, h], dtype=int)

        # Consistent integer PID mapping for query and gallery in evaluation
        test_df = splits_df[splits_df[split_col].isin(['query', 'gallery'])]
        test_identities = sorted(list(test_df['DOG_ID'].unique()))
        test_id_to_pid = {id_: idx for idx, id_ in enumerate(test_identities)}

        train = self._process_split(splits_df, split_col, 'train', bboxes_dict, test_id_to_pid)
        query = self._process_split(splits_df, split_col, 'query', bboxes_dict, test_id_to_pid)
        gallery = self._process_split(splits_df, split_col, 'gallery', bboxes_dict, test_id_to_pid)

        logger.info(
            f"Loaded {self.__class__.__name__} ({self.split_type}-set, img_load={self.img_load}): "
            f"{len(train)} train, {len(query)} query, {len(gallery)} gallery images."
        )

        super(DogReID, self).__init__(train, query, gallery, **kwargs)

    def _process_split(self, splits_df, split_col, split, bboxes_dict, test_id_to_pid):
        sub_df = splits_df[splits_df[split_col] == split]
        data = []
        missing_images = 0

        for _, row in sub_df.iterrows():
            dog_id = str(row['DOG_ID'])
            video_id = str(row['VIDEO_ID'])
            img_path = osp.join(self.images_dir, dog_id, f"{dog_id}-{video_id}.jpg")

            if not osp.exists(img_path):
                missing_images += 1
                continue

            if split == 'train':
                pid = f"{self.dataset_name}_{dog_id}"
                camid = 0
            else:
                pid = test_id_to_pid[dog_id]
                camid = 1 if split == 'query' else 2

            if self.img_load == 'bbox':
                bbox = bboxes_dict.get((dog_id, video_id), None)
            else:
                bbox = None

            data.append((img_path, pid, camid, bbox))

        if missing_images > 0:
            logger.warning(f"Split {split}: {missing_images} images not found on disk!")

        return data


@DATASET_REGISTRY.register()
class DogReIDOpen(DogReID):
    """DogReID Open-set split with bbox cropping."""
    dataset_name = "DogReIDOpen"

    def __init__(self, root='datasets', **kwargs):
        super(DogReIDOpen, self).__init__(root=root, split_type='open', img_load='bbox', **kwargs)


@DATASET_REGISTRY.register()
class DogReIDFull(DogReID):
    """DogReID Closed-set without bbox cropping (full image)."""
    dataset_name = "DogReIDFull"

    def __init__(self, root='datasets', **kwargs):
        super(DogReIDFull, self).__init__(root=root, split_type='closed', img_load='full', **kwargs)


@DATASET_REGISTRY.register()
class DogReIDOpenFull(DogReID):
    """DogReID Open-set without bbox cropping (full image)."""
    dataset_name = "DogReIDOpenFull"

    def __init__(self, root='datasets', **kwargs):
        super(DogReIDOpenFull, self).__init__(root=root, split_type='open', img_load='full', **kwargs)


# Register dash aliases so users can specify "DogReID-Closed" or "DogReID-Open" in YAML configs
DATASET_REGISTRY._do_register("DogReID-Closed", DogReID)
DATASET_REGISTRY._do_register("DogReID-Open", DogReIDOpen)
