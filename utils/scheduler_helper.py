from timm.scheduler import CosineLRScheduler

def get_lr_scheduler(optimizer, scheduler_config):
    
    return CosineLRScheduler(
        optimizer,
        T_max=scheduler_config.get('scheduler_t_max', 50),
        eta_min=scheduler_config.get('scheduler_eta_min', 0)
    )