import torch
from timm.optim import create_optimizer_v2
from torch.optim import Adam, AdamW

def get_optimizer(model, optimizer_config):
    lr = 1e-5
    weight_decay = 0.001 
    betas = (0.9, 0.999)
    eps = 1e-8

    return create_optimizer_v2(model.parameters(), opt='adamw', lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
