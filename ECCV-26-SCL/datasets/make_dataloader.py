import torch
import torch.distributed as dist
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
from torch.utils.data import BatchSampler, DataLoader

from .bases import CombineImageDataset, ImageDataset
from .combine import CombineDataset
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP


def train_collate_fn(batch):
    imgs, pids, camids, viewids, speids, names, marks, _ = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.int64),
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        torch.tensor(speids, dtype=torch.int64),
        names,
        torch.tensor(marks, dtype=torch.int64),
    )


def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        pids,
        camids,
        torch.tensor(camids, dtype=torch.int64),
        torch.tensor(viewids, dtype=torch.int64),
        img_paths,
    )


def _build_train_transforms(cfg):
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(
                probability=cfg.INPUT.RE_PROB,
                mode="pixel",
                max_count=1,
                device="cpu",
            ),
        ]
    )


def _build_val_transforms(cfg):
    return T.Compose(
        [
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )


def make_dataloader(cfg):
    if cfg.DATASETS.NAMES != "combine":
        raise ValueError("The clean MetaN release expects DATASETS.NAMES='combine'.")

    train_transforms = _build_train_transforms(cfg)
    val_transforms = _build_val_transforms(cfg)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_dataset = CombineDataset(
        names=cfg.DATASETS.TRAIN_COMBINE_NAMES,
        roots=cfg.DATASETS.TRAIN_ROOTS,
        combine_pid=cfg.DATASETS.COMBINE_PID,
        wildfile_names=cfg.DATASETS.WILDLIFE71.WILDLIFE_NAMES,
        train_split_mode=cfg.DATASETS.WILDLIFEREID10K.TRAIN_SPLIT_MODE,
    )
    train_set = CombineImageDataset(train_dataset.train, train_transforms)

    num_queries = train_dataset.num_queries
    num_images = train_dataset.num_marks
    _, _, cam_num, view_num, species_num = train_dataset.get_dataset_info(train_dataset.train)
    num_classes = train_dataset.num_classes

    if "triplet" in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(
                train_dataset.train,
                cfg.SOLVER.IMS_PER_BATCH,
                cfg.DATALOADER.NUM_INSTANCE,
            )
            batch_sampler = BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(
                    train_dataset.train,
                    cfg.SOLVER.IMS_PER_BATCH,
                    cfg.DATALOADER.NUM_INSTANCE,
                ),
                num_workers=num_workers,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
    elif cfg.DATALOADER.SAMPLER == "softmax":
        train_loader = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=train_collate_fn,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unsupported sampler: {cfg.DATALOADER.SAMPLER}")

    val_loaders = {}
    if cfg.TEST.EVAL:
        test_dataset = CombineDataset(
            names=cfg.DATASETS.TEST_COMBINE_NAMES,
            roots=cfg.DATASETS.TEST_ROOTS,
            combine_pid=cfg.DATASETS.COMBINE_PID,
            wildfile_names=None,
            train_split_mode="all",
        )
        for name in cfg.DATASETS.TEST_COMBINE_NAMES:
            key = name.lower()
            val_set = ImageDataset(test_dataset.query_dict[key], val_transforms)
            val_loaders[key] = DataLoader(
                val_set,
                batch_size=cfg.TEST.IMS_PER_BATCH,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=val_collate_fn,
                pin_memory=True,
            )

    return (
        train_loader,
        val_loaders,
        num_queries,
        num_classes,
        cam_num,
        view_num,
        species_num,
        num_images,
    )
