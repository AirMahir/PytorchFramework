from timm.scheduler import CosineLRScheduler

def get_lr_scheduler(optimizer):
    return CosineLRScheduler(optimizer ,t_initial=25, lr_min=1e-6)
