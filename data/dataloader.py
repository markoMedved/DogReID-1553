import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import DOGVideoREIDDataset
from pytorch_metric_learning.samplers import MPerClassSampler
from data.reid_transforms import build_video_transforms

def _worker_init_fn(worker_id):
    import torch
    import os
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def build_dataloaders(cfg):
    """Build the train and validation dataloaders for our experiments"""
    
    train_tf = build_video_transforms(cfg, is_train=True)
    eval_tf = build_video_transforms(cfg, is_train=False)

    # --- Shared Dataset Parameters ---
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "world": cfg.world,
        "mask_dog": getattr(cfg, "mask_dog", False),
        "force_yolo": getattr(cfg, "force_yolo", False),
        "bbox_file": getattr(cfg, "bbox_file", None),
    }

    # --- Base Training Dataset (SPLIT='train') ---
    # label_map=None -> contiguous labels over the training identities only, so the
    # classifier has #train-ID outputs (as in OpenAnimals), not one per dog in the dataset.
    base_train_dataset = DOGVideoREIDDataset(split="train", transform=train_tf, label_map=None, **dataset_kwargs)

    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    # --- Full-train mode: train on the whole train split, evaluate on the test split ---
    if getattr(cfg, "val_split", 0) <= 0:
        sampler = MPerClassSampler(
            labels=base_train_dataset.labels,
            m=cfg.k,
            batch_size=cfg.batch_size,
            length_before_new_iter=len(base_train_dataset),
        )
        train_loader = DataLoader(
            base_train_dataset, batch_size=cfg.batch_size, sampler=sampler,
            drop_last=True, **loader_kwargs
        )
        query_loader, gallery_loader = build_test_loaders(cfg)
        print(f"--- Data Loading Stats ---")
        print(f"Training (full train split): {len(base_train_dataset)} samples, "
              f"{len(base_train_dataset.id_map)} identities")
        print(f"Evaluation on TEST split: Query={len(query_loader.dataset)}, Gallery={len(gallery_loader.dataset)}")
        return train_loader, query_loader, gallery_loader

    base_val_dataset = DOGVideoREIDDataset(split="train", transform=eval_tf,
                                           label_map=base_train_dataset.id_map, **dataset_kwargs)

    # --- Split Dog IDs for Validation ---
    # Avoids identity leakage between training and validation sets
    unique_train_dog_ids = np.array(sorted(list(set(base_train_dataset.dog_ids))))
    
    np.random.seed(42)
    np.random.shuffle(unique_train_dog_ids)
    
    # Get the amount of validation ids, as specified in the config file
    val_id_count = int(len(unique_train_dog_ids) * cfg.val_split)
    val_dog_ids = set(unique_train_dog_ids[:val_id_count])

    # --- Collect Indices ---
    train_indices = []
    val_query_indices = []
    val_gallery_indices = []

    # --- Split Validation into Query/Gallery ---
    # Utilizes 'GROUP' logic to separate query vs. gallery samples
    for i in range(len(base_train_dataset)):
        dog_id = base_train_dataset.dog_ids[i]

        # Seperate into scence disjoint groups for query and gallery
        if dog_id in val_dog_ids:
            group_val = base_train_dataset.df.iloc[i]['GROUP']
            if group_val == 1:
                val_query_indices.append(i)
            else:
                val_gallery_indices.append(i)
        else:
            train_indices.append(i)

    # --- Create PyTorch Subsets ---
    train_dataset = Subset(base_train_dataset, train_indices)
    val_query_dataset = Subset(base_val_dataset, val_query_indices)
    val_gallery_dataset = Subset(base_val_dataset, val_gallery_indices)

    # --- PK Sampler Initialization ---
    # Ensures batches contain 'P' identities with 'K' clips each
    subset_labels = [base_train_dataset.labels[i] for i in train_indices]
    
    sampler = MPerClassSampler(
        labels=subset_labels,  
        m=cfg.k,
        batch_size=cfg.batch_size,
        length_before_new_iter=len(train_dataset) # Only use the length of the dataset
    )

    # --- Construct DataLoaders ---
    # persistent_workers avoids respawning workers every epoch
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        drop_last=True, **loader_kwargs
    )

    val_loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)
    if cfg.num_workers > 0:
        val_loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    # For validation we need query and gallery dataloaders
    val_batch_size = min(cfg.batch_size, 32)
    val_query_loader = DataLoader(
        val_query_dataset, batch_size=val_batch_size,
        shuffle=False, **val_loader_kwargs
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_dataset, batch_size=val_batch_size,
        shuffle=False, **val_loader_kwargs
    )

    print(f"--- Data Loading Stats ---")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: Query={len(val_query_dataset)}, Gallery={len(val_gallery_dataset)}")

    return train_loader, val_query_loader, val_gallery_loader


def build_test_loaders(cfg, query_images=False, gallery_images=False, images=None):
    """Test loaders supporting Image-to-Image, Image-to-Video, and Video-to-Video."""
    if images is not None:
        query_images = images
        gallery_images = images

    eval_tf = build_video_transforms(cfg, is_train=False)
    
    full_df = pd.read_csv(cfg.split_file)
    
    # --- Global DOG_ID Mapping ---
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # --- Shared Dataset Parameters ---
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "transform": eval_tf,
        "world": cfg.world,
        "label_map": global_id_map,
        "mask_dog": getattr(cfg, "mask_dog", False),
        "bbox_file": getattr(cfg, "bbox_file", None),
    }

    # --- Query Configuration ---
    query_kwargs = dataset_kwargs.copy()
    query_kwargs["use_videos"] = not query_images
    query_kwargs["clip_len"] = 1 if query_images else cfg.clip_len
    
    use_gt_query = getattr(cfg, "use_gt_for_query_mask", False)
    if query_images and use_gt_query:
        print("-> [INFO] Special Config Active: Query set will use Ground Truth boxes.")
        query_kwargs["force_yolo"] = False
    else:
        query_kwargs["force_yolo"] = getattr(cfg, "force_yolo", False)

    # --- Gallery Configuration ---
    gallery_kwargs = dataset_kwargs.copy()
    gallery_kwargs["use_videos"] = not gallery_images
    gallery_kwargs["clip_len"] = 1 if gallery_images else cfg.clip_len
    
    use_gt_gallery = getattr(cfg, "use_gt_for_gallery_mask", False)
    if gallery_images and use_gt_gallery:
        print("-> [INFO] Special Config Active: Gallery set will use Ground Truth boxes.")
        gallery_kwargs["force_yolo"] = False
    else:
        gallery_kwargs["force_yolo"] = getattr(cfg, "force_yolo", False)

    query_dataset = DOGVideoREIDDataset(
        split="query", 
        **query_kwargs
    )
    
    gallery_dataset = DOGVideoREIDDataset(
        split="gallery", 
        **gallery_kwargs
    )

    # --- Construct Test DataLoaders ---
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    # Capped at 32 clips: 128 clips x 8 frames x 384^2 per batch exhausts host RAM
    test_batch_size = min(cfg.batch_size, 32)
    query_loader = DataLoader(
        query_dataset, batch_size=test_batch_size,
        shuffle=False, **loader_kwargs
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=test_batch_size,
        shuffle=False, **loader_kwargs
    )

    # --- Print Stats & Mode ---
    q_mode = "Image" if query_images else "Video"
    g_mode = "Image" if gallery_images else "Video"
    
    print(f"--- Test Loaders Ready ---")
    print(f"Mode: {q_mode}-to-{g_mode}")
    print(f"Query: {len(query_dataset)} | Gallery: {len(gallery_dataset)}")
    print(f"Background Masking Baseline: {dataset_kwargs['mask_dog']}")

    return query_loader, gallery_loader