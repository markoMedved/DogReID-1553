import torch
from pathlib import Path

class Config:


    data_root =  Path(__file__).resolve().parent.parent
    split_file = data_root / "splits.csv"

    output_dir = "trained_models"

    world = "closed"

    batch_size = 8

    num_ids = 4
    num_instances = 2
    
    num_workers = 4
    clip_len = 16

    model = "resnet"
    backbone = "resnet50"
    embedding_dim = 512


    epochs = 1
    lr = 3e-4
    weight_decay = 1e-5

    eval_freq = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"