import os
import argparse
import torch
import torch.nn as nn
import logging
import random
import timm
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, Dataset
from datasets.classification_dataset import ClassificationData
from datasets.segmentation_dataset import SegmentationData
from utils.transforms import train_transforms_classification, val_transforms_classification, train_transform_segmentation, val_transform_segmentation
from utils.helpers import read_config, get_device, generate_dirs
from trainers.classification_trainer import ClassificationTrainer
from trainers.segmentation_trainer import SegmentationTrainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True

def setup_logger(log_file):
    logging.basicConfig(
        filename = log_file,
        encoding = "utf-8",
        level=logging.DEBUG,
        format = '%(levelname)s - %(message)s'
    )

    return logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    parser.add_argument("--checkpoint", type = str, help = "Path to the checkpoint model - state dict")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    # get_seed(configs.get('seed', 42))
    device = get_device()

    logger = setup_logger(os.path.join(configs["output_dir"], 'log.txt'))
    logger.info("Starting main processing")

    logger.info(f"Using device: {device}")

    data_dir = configs['data_dir']
    generate_dirs(configs)

    if(configs['task_type'] == '0'):

        logger.info("Classification training....")

        train_data = ClassificationData(os.path.join(data_dir, "classificationData", "train"), transform = train_transforms_classification)
        test_data = ClassificationData(os.path.join(data_dir, "classificationData", "test"), transform = val_transforms_classification)

        train_dataloader = DataLoader(train_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = True, drop_last=True, pin_memory=True, persistent_workers=True)
        test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = False, pin_memory=True, persistent_workers=True)
        steps_per_epoch = len(train_dataloader)
        
        model = timm.create_model('resnet50d', pretrained=False, num_classes=3)
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

        trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, device, configs, logger=None)

        results = trainer.train()
        logger.info(results)
        # save_metrics(results)

    else:
        train_dataset = SegmentationData(os.path.join(data_dir, "segmentationData", "train"), transform = train_transform_segmentation)
        test_dataset = SegmentationData(os.path.join(data_dir, "segmentationData", "test"), transform = val_transform_segmentation)

        train_dataloader = DataLoader(train_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = True, drop_last=True, pin_memory=True, persistent_workers=True)
        test_dataloader = DataLoader(test_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = False, pin_memory=True, persistent_workers=True)

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
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
        # save_metrics(results)


if __name__ == "__main__":
    main()