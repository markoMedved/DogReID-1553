import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import DOGVideoREIDDataset
from .transforms import VideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):
    transform = VideoTransform()

    # 1. Map IDs globally for consistency
    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # 2. Setup common kwargs (Matching your Dataset class exactly)
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "world": cfg.world,
        "label_map": global_id_map
    }

    # 3. Initialize the BASE Training Dataset
    # This filters by cfg.world internally to find 'train' samples
    base_train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)

    # 4. Create Leakage-Free Validation Split
    # We use indices based on the length of the filtered base_train_dataset
    indices = np.arange(len(base_train_dataset))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = int(len(base_train_dataset) * cfg.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # 5. Create Subsets
    train_dataset = Subset(base_train_dataset, train_indices)
    val_subset = Subset(base_train_dataset, val_indices)

    # 6. PK Sampler (Extract labels from the base dataset for the subset)
    subset_labels = [base_train_dataset.labels[i] for i in train_indices]
    
    sampler = MPerClassSampler(
        labels=subset_labels,  
        m=cfg.k,
        batch_size=cfg.batch_size,
        length_before_new_iter=len(train_dataset) 
    )

    # 7. Final Loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        sampler=sampler, 
        drop_last=True, 
        num_workers=cfg.num_workers
    )

    # Split the validation subset into Query and Gallery (50/50)
    mid = len(val_subset) // 2
    val_query_loader = DataLoader(
        Subset(val_subset, range(0, mid)), 
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers
    )
    val_gallery_loader = DataLoader(
        Subset(val_subset, range(mid, len(val_subset))), 
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers
    )

    print(f"--- Data Loading Stats (Validation Mode) ---")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_subset)}")

    return train_loader, val_query_loader, val_gallery_loader


def build_test_loaders(cfg):
    """Clean Test Loaders for Final Evaluation using actual CSV splits."""
    transform = VideoTransform()
    full_df = pd.read_csv(cfg.split_file)
    
    # Global map must be the same as training
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "world": cfg.world,
        "label_map": global_id_map
    }

    query_dataset = DOGVideoREIDDataset(split="query", **dataset_kwargs)
    gallery_dataset = DOGVideoREIDDataset(split="gallery", **dataset_kwargs)

    query_loader = DataLoader(
        query_dataset, batch_size=cfg.batch_size * 2, 
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    print(f"--- Test Loaders Ready ---")
    print(f"Query: {len(query_dataset)} | Gallery: {len(gallery_dataset)}")

    return query_loader, gallery_loader