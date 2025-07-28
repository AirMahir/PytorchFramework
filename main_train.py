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


def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device()

    data_dir = configs['data_dir']
    generate_dirs()

    if(configs['task_type'] == '0'):

        train_transforms, test_transforms =  get_classification_transforms()

        train_data = ClassificationData(os.path.join(data_dir, "classificationData\train"), transform=train_transforms)
        test_data = ClassificationData(os.path.join(data_dir, "classificationData\test"), transform=test_transforms)

        train_dataloader = DataLoader(train_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = True)
        test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = False)

        model = timm.create_model('resnet50d', pretrained=True, num_classes=3)

        criterion = nn.Softmax()

        optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'])

        trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer , criterion, device, configs, logger=None)

        results = trainer.train()

        save_metrics(results)

    else:
        train_dataset = SimpleOxfordPetDataset(os.path.join(data_dir, "segmentationData"), "train")
        val_dataset = SimpleOxfordPetDataset(os.path.join(data_dir, "segmentationData"), "valid")


        train_dataloader = DataLoader(train_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = True)
        valid_dataloader = DataLoader(val_dataset, configs['batch_size'], num_workers=configs['num_workers'], shuffle = False)

        model = smp.create_model(
            "FPN",
            encoder_name="resnet34",
            in_channels=3,
            classes=1
        )

