import torch

class Config:
    model_name = "model06_pruebaGP_15epochs_6iters_lr005_batchsize256"
    n_iters = 6
    batch_size = 64*4
    num_classes = 10
    lr = 0.05
    num_epochs = 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
