import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from .dataset import DOGVideoREIDDataset
from .transforms import VideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):
    transform = VideoTransform()

    # global DOG_ID mapping
    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    # shared dataset parameters
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "world": cfg.world,
        "label_map": global_id_map
    }

    # base dataset (SPLIT='train')
    base_train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)

    # split dog IDs for validation (avoid identity leakage)
    unique_train_dog_ids = np.array(sorted(list(set(base_train_dataset.dog_ids))))
    
    np.random.seed(42)
    np.random.shuffle(unique_train_dog_ids)
    
    val_id_count = int(len(unique_train_dog_ids) * cfg.val_split)
    val_dog_ids = set(unique_train_dog_ids[:val_id_count])

    # collect indices
    train_indices = []
    val_query_indices = []
    val_gallery_indices = []

    # split validation into query/gallery using GROUP
    for i in range(len(base_train_dataset)):
        dog_id = base_train_dataset.dog_ids[i]

        if dog_id in val_dog_ids:
            group_val = base_train_dataset.df.iloc[i]['GROUP']
            if group_val == 1:
                val_query_indices.append(i)
            else:
                val_gallery_indices.append(i)
        else:
            train_indices.append(i)

    # create subsets
    train_dataset = Subset(base_train_dataset, train_indices)
    val_query_dataset = Subset(base_train_dataset, val_query_indices)
    val_gallery_dataset = Subset(base_train_dataset, val_gallery_indices)

    # PK sampler (P IDs × K clips)
    subset_labels = [base_train_dataset.labels[i] for i in train_indices]
    
    sampler = MPerClassSampler(
        labels=subset_labels,  
        m=cfg.k,
        batch_size=cfg.batch_size,
        length_before_new_iter=len(train_dataset) 
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler, 
        drop_last=True, num_workers=cfg.num_workers
    )

    val_query_loader = DataLoader(
        val_query_dataset, batch_size=cfg.batch_size * 2, 
        shuffle=False, num_workers=cfg.num_workers
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_dataset, batch_size=cfg.batch_size * 2, 
        shuffle=False, num_workers=cfg.num_workers
    )

    print(f"--- Data Loading Stats ---")
    print(f"Training: {len(train_dataset)} samples")
    print(f"Validation: Query={len(val_query_dataset)}, Gallery={len(val_gallery_dataset)}")

    return train_loader, val_query_loader, val_gallery_loader


def build_test_loaders(cfg):
    """Test loaders using CSV splits."""
    transform = VideoTransform()
    full_df = pd.read_csv(cfg.split_file)
    
    # same global ID map
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

    # query/gallery datasets
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