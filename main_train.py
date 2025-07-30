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
from utils.helpers import read_config, get_device, generate_dirs
from trainers.classification_trainer import ClassificationTrainer
from trainers.segmentation_trainer import SegmentationTrainer
from utils.transforms import train_transforms_classification, val_transforms_classification, train_transform_segmentation, val_transform_segmentation

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

def setup_logger(log_file):
    logging.basicConfig(
        filename = log_file,
        encoding = "utf-8",
        level=logging.INFO, # Changed to INFO for a more professional log, can be DEBUG if needed
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Also adding a console handler to see logs in terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    # Get the root logger and add the console handler
    root_logger = logging.getLogger()
    if not root_logger.handlers: # Avoid adding duplicate handlers if setup_logger is called multiple times
        root_logger.addHandler(console_handler)
    return logging.getLogger(__name__)

def run_classification_training(configs, device, logger, data_dir):
    logger.info("Classification Training")

    train_data = ClassificationDataset(os.path.join(data_dir, "classificationData", "train"), transform = train_transforms_classification)
    test_data = ClassificationDataset(os.path.join(data_dir, "classificationData", "test"), transform = val_transforms_classification)

    train_dataloader = DataLoader(train_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = True, drop_last=True, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = False, pin_memory=True, persistent_workers=True)
    steps_per_epoch = len(train_dataloader)
    
    model = timm.create_model(configs['classification_config']['model_name'], pretrained=False, num_classes=configs['num_classes'])
    model.to(device)

    epochs = configs["num_epochs"]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=configs["learning_rate"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, device, configs, logger=logger)
    results = trainer.train()
    logger.info(results)


def run_segmentation_training(configs, device, logger, data_dir):
    train_dataset = SegmentationDataset(os.path.join(data_dir, "segmentationData", "train"), transform = train_transform_segmentation)
    test_dataset = SegmentationDataset(os.path.join(data_dir, "segmentationData", "test"), transform = val_transform_segmentation)

    train_dataloader = DataLoader(train_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = True, drop_last=True, pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = False, pin_memory=True, persistent_workers=True)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=configs['num_classes']
    )
    model.to(device)

    # Expects (B, 1, H, W )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=configs['learning_rate'])
    steps_per_epoch = len(train_dataloader)
    epochs = configs["num_epochs"]

    if steps_per_epoch == 0 or epochs == 0:
        raise ValueError("Invalid scheduler configuration: steps_per_epoch or epochs is 0.")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=configs["learning_rate"],
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )

    trainer = SegmentationTrainer(model, train_dataloader, test_dataloader, optimizer , criterion, scheduler, device, configs, logger=logger)

    results = trainer.train()
    logger.info(results)

def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--checkpoint", type = str, help = "Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()

    logger = setup_logger(os.path.join(configs["output_dir"], 'log.txt'))
    logger.info("Starting the training script")
    logger.info(f"Using device: {device}")

    data_dir = configs['data_dir']
    generate_dirs(configs)

    if(configs['task_type'] == '0'):
        run_classification_training(configs, device, logger, data_dir)

    else:
        run_segmentation_training(configs, device, logger, data_dir)

if __name__ == "__main__":
    main()