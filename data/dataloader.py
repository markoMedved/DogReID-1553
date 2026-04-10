from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import VideoTransforms

def build_dataloaders(cfg):

    transform = VideoTransforms()

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    return train_loader, query_loader, gallery_loader