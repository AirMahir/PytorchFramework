import os
import timm
import torch
import logging
import argparse
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets.classification_dataset import ClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from utils.helpers import read_config, get_device, generate_dirs, seed_everything, set_pytorch_optimizations
from trainers.classification_trainer import ClassificationTrainer
from trainers.segmentation_trainer import SegmentationTrainer
from utils.transforms import train_transforms_classification, val_transforms_classification, train_transform_segmentation, val_transform_segmentation
from utils.logger import setup_logger
from utils.optimizer_helper import get_optimizer
from utils.scheduler_helper import get_lr_scheduler

# Set PyTorch performance optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
set_pytorch_optimizations()

def run_classification_training(configs, device, logger, checkpoint_path):
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
    
    logger.info(f"Loaded {len(train_data)} training samples and {len(test_data)} validation samples.")
    logger.info(f"Using {len(train_dataloader)} training batches and {len(test_dataloader)} validation batches.")

    model = timm.create_model(model_cfg['name'], pretrained=model_cfg.get('pretrained', False), num_classes=model_cfg['num_classes'])
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = get_optimizer(model, opt_cfg)
    scheduler = get_lr_scheduler(optimizer)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    start_epoch = 0
    if checkpoint_path:
        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch-1}")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info("Model checkpoint loaded successfully.")

    trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, scaler, device, configs, logger=logger, start_epoch=start_epoch)
    results = trainer.train()
    logger.info(results)

def run_segmentation_training(configs, device, logger, checkpoint_path=None):
    logger.info("Segmentation Training")
    data_cfg = configs['data']
    model_cfg = configs['model']
    train_cfg = configs['training']
    opt_cfg = configs['optimizer_type']
    sched_cfg = configs['scheduler_type']

    train_dataset = SegmentationDataset(data_cfg['train_dir'], transform=train_transform_segmentation)
    test_dataset = SegmentationDataset(data_cfg['val_dir'], transform=val_transform_segmentation)

    train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'], shuffle=False, pin_memory=True, persistent_workers=True)

    # display_segmentation_batch(train_dataset.images, train_dataset.masks, 0, configs, class_map=train_dataset.class_map, n=8)
   
    model = smp.Unet(
        encoder_name=model_cfg['encoder_name'],
        encoder_weights=model_cfg['encoder_weights'],
        in_channels=model_cfg['in_channels'],
        classes=model_cfg['classes']
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = get_optimizer(model, opt_cfg)
    scheduler = get_lr_scheduler(optimizer)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision

    start_epoch = 0
    if checkpoint_path:
        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch-1}")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info("Model checkpoint loaded successfully.")

    trainer = SegmentationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, scaler, device, configs, logger=logger, start_epoch=start_epoch)
    results = trainer.train()
    logger.info(results)

def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--checkpoint", type = str, help = "Path to the checkpoint model")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()
    generate_dirs(configs)
    seed_everything(configs["seed"])

    logger = setup_logger(os.path.join(configs["output_dir"], 'log.txt'))
    logger.info("Starting the training script")
    logger.info(f"Using device: {device}")

    if(configs['task_type'] == 'classification'):
        run_classification_training(configs, device, logger, args.checkpoint)

    else:
        run_segmentation_training(configs, device, logger, args.checkpoint)

if __name__ == "__main__":
    main()