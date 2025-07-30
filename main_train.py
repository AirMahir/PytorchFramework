import os
import timm
import torch
import logging
import argparse
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader
from datasets.classification_dataset import ClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from utils.helpers import read_config, get_device, generate_dirs, seed_everything
from trainers.classification_trainer import ClassificationTrainer
from trainers.segmentation_trainer import SegmentationTrainer
from utils.transforms import train_transforms_classification, val_transforms_classification, train_transform_segmentation, val_transform_segmentation
from utils.logger import setup_logger
from timm.optim import AdamP
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

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


def run_classification_training(configs, device, logger, data_dir):
    logger.info("Classification Training")
    data_cfg = configs['data']
    model_cfg = configs['model']
    train_cfg = configs['training']
    opt_cfg = configs['optimizer']
    sched_cfg = configs['scheduler']

    train_data = ClassificationDataset(data_cfg['train_dir'], transform=train_transforms_classification, logger=logger)
    test_data = ClassificationDataset(data_cfg['val_dir'], transform=val_transforms_classification, logger=logger)

    train_dataloader = DataLoader(train_data, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=False, pin_memory=True, persistent_workers=True)
    steps_per_epoch = len(train_dataloader)

    model = timm.create_model(model_cfg['name'], pretrained=model_cfg.get('pretrained', False), num_classes=model_cfg['num_classes'])
    model.to(device)

    epochs = train_cfg['epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, opt_cfg)
    scheduler = get_scheduler(optimizer, sched_cfg)

    trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, device, configs, logger=logger)
    results = trainer.train()
    logger.info(results)


def run_segmentation_training(configs, device, logger, data_dir):
    logger.info("Segmentation Training")
    data_cfg = configs['data']
    model_cfg = configs['model']
    train_cfg = configs['training']
    opt_cfg = configs['optimizer']
    sched_cfg = configs['scheduler']

    train_dataset = SegmentationDataset(data_cfg['train_dir'], transform=train_transform_segmentation)
    test_dataset = SegmentationDataset(data_cfg['val_dir'], transform=val_transform_segmentation)

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=False, pin_memory=True, persistent_workers=True)

    model = smp.Unet(
        encoder_name=model_cfg['encoder_name'],
        encoder_weights=model_cfg['encoder_weights'],
        in_channels=model_cfg['in_channels'],
        classes=model_cfg['classes']
    )
    model.to(device)

    epochs = train_cfg['epochs']
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, opt_cfg)
    scheduler = get_scheduler(optimizer, sched_cfg)

    trainer = SegmentationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, device, configs, logger=logger)
    results = trainer.train()
    logger.info(results)

def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--checkpoint", type = str, help = "Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()
    generate_dirs(configs)
    seed_everything(configs["seed"])

    logger = setup_logger(os.path.join(configs["output_dir"], 'log.txt'))
    logger.info("Starting the training script")
    logger.info(f"Using device: {device}")

    data_dir = configs['data_dir']

    if(configs['task_type'] == '0'):
        run_classification_training(configs, device, logger, data_dir)

    else:
        run_segmentation_training(configs, device, logger, data_dir)

if __name__ == "__main__":
    main()