import torch
from pathlib import Path

class Config:


    data_root =  Path(__file__).resolve().parent.parent
    split_file = data_root / "splits.csv"

    output_dir = "trained_models_vit" # Change if change model
    model = "vit" # Change if change model

    world = "closed"

    batch_size = 4

    num_ids = 4
    num_instances = 2
    
    num_workers = 4
    clip_len = 1 # TODO CHANGE after debugging

    embedding_dim = 512

    epochs = 50
    lr = 3e-4
    weight_decay = 1e-5
    eval_period = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"

        
    # ---- evaluation mode ----
    eval_only = True
    checkpoint_path = "output/best_model.pth"