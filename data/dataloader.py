import pandas as pd
from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import VideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):
    transform = VideoTransform()

    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": global_id_map
    }

    # All these use split="train" from your CSV
    # We use these for internal validation during training
    train_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)
    val_query_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)
    val_gallery_dataset = DOGVideoREIDDataset(split="train", **dataset_kwargs)

    # Note: You can add a simple index filter in your Dataset __init__ 
    # to make sure val_query and val_gallery don't overlap perfectly.

    sampler = MPerClassSampler(
        labels=train_dataset.labels,  
        m=cfg.k,
        batch_size=cfg.batch_size,
        length_before_new_iter=len(train_dataset) 
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler, 
        drop_last=True, num_workers=cfg.num_workers
    )

    # Validation loaders built from the "Train" split data
    val_query_loader = DataLoader(
        val_query_dataset, batch_size=cfg.batch_size * 2, 
        shuffle=False, num_workers=cfg.num_workers
    )
    
    val_gallery_loader = DataLoader(
        val_gallery_dataset, batch_size=cfg.batch_size * 2,
        shuffle=False, num_workers=cfg.num_workers
    )

    return train_loader, val_query_loader, val_gallery_loader



def build_test_loaders(cfg):
    transform = VideoTransform()

    full_df = pd.read_csv(cfg.split_file)
    all_unique_ids = sorted(full_df["DOG_ID"].unique())
    global_id_map = {dog_id: i for i, dog_id in enumerate(all_unique_ids)}

    dataset_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": global_id_map
    }

    # Standard Re-ID evaluation splits
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