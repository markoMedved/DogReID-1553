from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import ViTVideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):
    # 1. Initialize Transform
    transform = ViTVideoTransform()

    # 2. Build Training Set First
    train_dataset = DOGVideoREIDDataset(
        root_dir=cfg.data_root,
        split_file=cfg.split_file,
        split="train",
        world=cfg.world,
        clip_len=cfg.clip_len,
        transform=transform
    )

    # 3. Build Eval Sets using the SAME label mapping as Train
    # This ensures "Dog_01" is ID 5 in all loaders.
    eval_kwargs = {
        "root_dir": cfg.data_root,
        "split_file": cfg.split_file,
        "world": cfg.world,
        "clip_len": cfg.clip_len,
        "transform": transform,
        "label_map": train_dataset.id_map  
    }

    query_dataset = DOGVideoREIDDataset(split="query", **eval_kwargs)
    gallery_dataset = DOGVideoREIDDataset(split="gallery", **eval_kwargs)

    # 4. Configure Sampler (P=4, K=2 for batch_size=8)
    sampler = MPerClassSampler(
        labels=train_dataset.labels,  
        m=2          
    )

    # 5. Build Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        sampler=sampler,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True # Speed up transfer to GPU
    )

    # Query and Gallery can have larger batch sizes for speed
    query_loader = DataLoader(
        query_dataset,
        batch_size=16, 
        shuffle=False,
        num_workers=cfg.num_workers
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    return train_loader, query_loader, gallery_loader