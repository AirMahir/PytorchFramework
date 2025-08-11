from timm.scheduler import CosineLRScheduler

def get_lr_scheduler(optimizer):
    return CosineLRScheduler(optimizer ,t_initial=50, lr_min=1e-5)
