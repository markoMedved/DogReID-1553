import torch

class Config:

    data_root = "C:/Users/marko/Desktop/DogReID-1553"
    split_file = "C:/Users/marko/Desktop/DogReID-1553/splits.csv"

    output_dir = "trained_models"

    world = "closed"

    batch_size = 8
    num_workers = 4
    clip_len = 16

    model = "resnet"
    backbone = "resnet50"
    embedding_dim = 512


    epochs = 5
    lr = 3e-4
    weight_decay = 1e-5

    eval_freq = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"