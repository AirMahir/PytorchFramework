from timm.scheduler import CosineLRScheduler

def get_lr_scheduler(optimizer):
    return CosineLRScheduler(optimizer ,t_initial=25, lr_min=0, t_in_epochs=True, cycle_decay=0.5,
                                                     warmup_t = 5, warmup_lr_init=1e-4, cycle_limit =1)
