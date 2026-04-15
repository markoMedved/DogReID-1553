from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import ViTVideoTransform
from pytorch_metric_learning.samplers import MPerClassSampler

def build_dataloaders(cfg):

    transform = ViTVideoTransform()

    train_dataset = DOGVideoREIDDataset(
        root_dir=cfg.data_root,
        split_file=cfg.split_file,
        split="train",
        world=cfg.world,
        clip_len=cfg.clip_len,
        transform=transform
    )

    query_dataset = DOGVideoREIDDataset(
        root_dir=cfg.data_root,
        split_file=cfg.split_file,
        split="query",
        world=cfg.world,
        clip_len=cfg.clip_len,
        transform=transform
    )

    gallery_dataset = DOGVideoREIDDataset(
        root_dir=cfg.data_root,
        split_file=cfg.split_file,
        split="gallery",
        clip_len=cfg.clip_len,
        transform=transform
    )

    sampler = MPerClassSampler(
        labels=train_dataset.labels,  
        m=2,
        length_before_new_iter=1000            
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,     # P*K
        sampler=sampler,
        drop_last=True,
        num_workers=4
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=1,       
        shuffle=False
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=1,
        shuffle=False
    )

    return train_loader, query_loader, gallery_loader