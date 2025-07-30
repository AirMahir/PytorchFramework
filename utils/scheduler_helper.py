from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

def get_scheduler(optimizer, scheduler_config):
    sched_type = scheduler_config.get('scheduler_type', 'None')
    if sched_type is None or sched_type.lower() == 'none':
        return None
    sched_type = sched_type.lower()
    if sched_type == 'cosineannealinglr':
        return CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('scheduler_t_max', 50),
            eta_min=scheduler_config.get('scheduler_eta_min', 0)
        )
    elif sched_type == 'steplr':
        return StepLR(
            optimizer,
            step_size=scheduler_config.get('scheduler_step_size', 10),
            gamma=scheduler_config.get('scheduler_gamma', 0.1)
        )
    elif sched_type == 'reducelronplateau':
        return ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('scheduler_mode', 'min'),
            factor=scheduler_config.get('scheduler_factor', 0.1),
            patience=scheduler_config.get('scheduler_patience', 10),
            threshold=scheduler_config.get('scheduler_threshold', 1e-4),
            cooldown=scheduler_config.get('scheduler_cooldown', 0),
            min_lr=scheduler_config.get('scheduler_min_lr', 0)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")