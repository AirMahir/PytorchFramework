import torch
from timm.optim import AdamP
from torch.optim import Adam, AdamW

def get_optimizer(model, optimizer_config):
    opt_type = optimizer_config.get('optimizer_type', 'AdamW').lower()
    lr = optimizer_config.get('learning_rate', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    betas = (
        optimizer_config.get('adamw_beta1', 0.9),
        optimizer_config.get('adamw_beta2', 0.999)
    )
    eps = optimizer_config.get('adamw_eps', 1e-8)
    if opt_type == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif opt_type == 'adamp':
        return AdamP(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")