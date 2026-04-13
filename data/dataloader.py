from torch.utils.data import DataLoader
from .dataset import DOGVideoREIDDataset
from .transforms import ViTVideoTransform
from samplers.sampler import RandomIdentitySampler

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

    sampler = RandomIdentitySampler(
        train_dataset,
        num_ids=cfg.num_ids,          # e.g. 4
        num_instances=cfg.num_instances  # e.g. 2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,   # must = num_ids * num_instances
        sampler=sampler,
        num_workers=4,
        drop_last=True
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