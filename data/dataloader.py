import pandas as pd
from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import ViTVideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):
    # 1. Initialize Transform (handles NumPy -> PIL -> Tensor)
    transform = ViTVideoTransform()

    # 2. Build a GLOBAL label map from the entire CSV
    # This prevents the "ID Collapse" where all test dogs become label -1
    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}
    
    print(f"--- Data Loading Stats ---")
    print(f"Total unique dogs in dataset: {len(global_id_map)}")

    # 3. Define common arguments
    # We pass the global_id_map to ensure consistency across all splits
    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": global_id_map
    }

    # 4. Initialize Datasets
    # Note: Ensure your Dataset class only filters counts > 1 if split == "train"
    train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)
    query_dataset = DOGVideoREIDDataset(split="query", **dataset_kwargs)
    gallery_dataset = DOGVideoREIDDataset(split="gallery", **dataset_kwargs)

    print(f"Train samples: {len(train_dataset)}  | Unique IDs: {len(set(train_dataset.labels))}")
    print(f"Query samples: {len(query_dataset)}  | Unique IDs: {len(set(query_dataset.labels))}")
    print(f"Gallery samples: {len(gallery_dataset)}| Unique IDs: {len(set(gallery_dataset.labels))}")

    sampler = MPerClassSampler(
            labels=train_dataset.labels,  
            m=2,
            batch_size=cfg.batch_size,
            # This is the magic line: 
            # It forces the epoch to end after we've seen each video roughly once.
            length_before_new_iter=len(train_dataset) 
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler, 
        drop_last=True,
        num_workers=cfg.num_workers
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.batch_size * 2, 
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_loader, query_loader, gallery_loader