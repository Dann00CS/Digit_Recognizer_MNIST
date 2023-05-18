import torch

class Config:
    model_name = "model05_1iter_28_28_toCompare"
    n_iters = 1
    batch_size = 64*2
    num_classes = 10
    lr = 0.01
    num_epochs = 30
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
