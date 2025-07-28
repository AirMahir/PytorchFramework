import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import timm
import segmentation_models_pytorch as smp

from torch.utils.data import DataLoader, Dataset
from datasets.classification_dataset import ClassificationData
from datasets.segmentation_dataset import SimpleOxfordPetDataset
from utils.transforms import get_classification_transforms, get_segmentation_transforms
from utils.helpers import read_config, get_seed, get_device, generate_dirs, save_metrics
from trainers.classification_trainer import ClassificationTrainer
from trainers.segmentation_trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    get_seed(configs.get('seed', 42))
    device = get_device()

    print(f"Using device: {device}")

    data_dir = configs['data_dir']
    generate_dirs()

    if(configs['task_type'] == '0'):

        train_transforms, test_transforms =  get_classification_transforms()

        train_data = ClassificationData(os.path.join(data_dir, "classificationData", "train"), transform=train_transforms)
        test_data = ClassificationData(os.path.join(data_dir, "classificationData", "test"), transform=test_transforms)

        train_dataloader = DataLoader(train_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = True)
        test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = False)

        model = timm.create_model('resnet50d', pretrained=True, num_classes=3)
        model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

        trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer , criterion, device, configs, logger=None)

        results = trainer.train()

        print(results)

        save_metrics(results)

    else:
        train_dataset = SimpleOxfordPetDataset(os.path.join(data_dir, "segmentationData"), "train")
        test_dataset = SimpleOxfordPetDataset(os.path.join(data_dir, "segmentationData"), "test")

        train_dataloader = DataLoader(train_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = True)
        test_dataloader = DataLoader(test_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = False)

        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
        model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=configs['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train_dataloader))

        trainer = SegmentationTrainer(model, train_dataloader, test_dataloader, optimizer , criterion, device, configs, logger=None)

        results = trainer.train()
        print(results)
        save_metrics(results)


