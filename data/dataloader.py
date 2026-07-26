import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import DOGVideoREIDDataset
from pytorch_metric_learning.samplers import MPerClassSampler
from data.reid_transforms import build_video_transforms

def build_dataloaders(cfg):
    """Build the train and validation dataloaders for our experiments"""
    
    train_tf = build_video_transforms(cfg, is_train=True)

    # --- Global DOG_ID Mapping ---
    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # --- Shared Dataset Parameters ---
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "transform": train_tf,
        "world": cfg.world,
        "label_map": global_id_map
    }

    # --- Base Training Dataset (SPLIT='train') ---
    base_train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)

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
    val_query_dataset = Subset(base_train_dataset, val_query_indices)
    val_gallery_dataset = Subset(base_train_dataset, val_gallery_indices)

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
    # persistent_workers avoids respawning workers every epoch, which is
    # expensive here because each one holds a YOLO detector
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        drop_last=True, **loader_kwargs
    )

    # For validation we need query and gallery dataloaders
    val_query_loader = DataLoader(
        val_query_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, **loader_kwargs
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, **loader_kwargs
    )

    print(f"--- Data Loading Stats ---")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: Query={len(val_query_dataset)}, Gallery={len(val_gallery_dataset)}")

    return train_loader, val_query_loader, val_gallery_loader


def build_test_loaders(cfg, images=False):
    """Test loaders using CSV splits."""
    eval_tf  = build_video_transforms(cfg, is_train=False)
    
    full_df = pd.read_csv(cfg.split_file)
    
    # --- Global DOG_ID Mapping ---
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # --- Shared Dataset Parameters ---
    # clip_len belongs here: leaving it out made the gallery fall back to the
    # dataset default instead of cfg.clip_len, so query and gallery clips were
    # different lengths.
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "transform": eval_tf,
        "world": cfg.world,
        "label_map": global_id_map
    }

    # --- Query and Gallery Datasets ---
    # The gallery is always video; only the query modality varies.
    query_kwargs = {**dataset_kwargs, "clip_len": 1 if images else cfg.clip_len}
    query_dataset = DOGVideoREIDDataset(
            split="query",
            use_videos=not images,
            **query_kwargs
        )

    gallery_dataset = DOGVideoREIDDataset(split="gallery", **dataset_kwargs)

    # --- Construct Test DataLoaders ---
    loader_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    query_loader = DataLoader(
        query_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, **loader_kwargs
    )

    gallery_loader = DataLoader(
        gallery_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, **loader_kwargs
    )

    print(f"--- Test Loaders Ready ---")
    print(f"Query: {len(query_dataset)} | Gallery: {len(gallery_dataset)}")

    return query_loader, gallery_loader