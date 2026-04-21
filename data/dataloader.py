import pandas as pd
from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import VideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
# ... your other imports

def build_dataloaders(cfg):
    transform = VideoTransform()
    full_df = pd.read_csv(cfg.split_file)

    # 1. Dynamically select the correct split column
    # If world is 'closed', use 'SPLIT_CLOSED_SET'. If 'open', use 'SPLIT_OPEN_SET'
    split_col = "SPLIT_CLOSED_SET" if cfg.world == "closed" else "SPLIT_OPEN_SET"
    
    if split_col not in full_df.columns:
        raise KeyError(f"Column {split_col} not found in CSV. Available: {full_df.columns.tolist()}")

    # 2. Global ID Mapping
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # 3. Filter for 'train' rows using the dynamic column name
    train_df = full_df[full_df[split_col] == "train"].reset_index(drop=True)
    
    # Check if train_df is empty (common if 'train' is actually 'TRAIN' or similar)
    if len(train_df) == 0:
        actual_values = full_df[split_col].unique()
        print(f"⚠️ Warning: No 'train' rows found in {split_col}. Values present: {actual_values}")

    # 4. IDENTIFY VAL IDs (Held-out dogs)
    unique_train_dogs = train_df["DOG_ID"].unique()
    np.random.seed(42) 
    np.random.shuffle(unique_train_dogs)
    
    num_val = int(len(unique_train_dogs) * cfg.val_split)
    val_dog_ids = set(unique_train_dogs[:num_val])
    train_dog_ids = set(unique_train_dogs[num_val:])


    # 4. Create Indices for Subsets
    train_indices = train_df[train_df["DOG_ID"].isin(train_dog_ids)].index.tolist()
    val_indices = train_df[train_df["DOG_ID"].isin(val_dog_ids)].index.tolist()

    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": global_id_map
    }

    # 5. Build Datasets
    # We use one base dataset and wrap it in Subsets to ensure NO LEAKAGE
    base_train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)
    
    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(base_train_dataset, val_indices)

    # 6. PK Sampler (Important: uses the labels of the filtered train set)
    # We need to extract the labels specifically for the subset
    subset_labels = [base_train_dataset.labels[i] for i in train_indices]
    
    sampler = MPerClassSampler(
        labels=subset_labels,  
        m=cfg.k,
        batch_size=cfg.batch_size,
        length_before_new_iter=len(train_dataset) 
    )

    # 7. Final Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler, 
        drop_last=True, num_workers=cfg.num_workers
    )

    # For validation, we use the held-out dogs. 
    # To simulate Query/Gallery, we just split the val_dataset in half
    mid = len(val_dataset) // 2
    val_query_loader = DataLoader(
        Subset(val_dataset, range(0, mid)), 
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers
    )
    val_gallery_loader = DataLoader(
        Subset(val_dataset, range(mid, len(val_dataset))), 
        batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers
    )

    print(f"--- Leakage-Free Splits ---")
    print(f"Training on: {len(train_dog_ids)} dogs ({len(train_indices)} clips)")
    print(f"Validating on: {len(val_dog_ids)} dogs ({len(val_indices)} clips)")

    return train_loader, val_query_loader, val_gallery_loader



def build_test_loaders(cfg):
    transform = VideoTransform()
    full_df = pd.read_csv(cfg.split_file)
    
    # Use the same dynamic column selection
    split_col = "SPLIT_CLOSED_SET" if cfg.world == "closed" else "SPLIT_OPEN_SET"

    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": {dog_id: i for i, dog_id in enumerate(sorted(full_df["DOG_ID"].unique()))},
        "split_column": split_col # Ensure your Dataset class knows which column to look at!
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

    return query_loader, gallery_loader