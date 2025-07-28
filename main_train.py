import argparse
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader, Dataset
from datasets.classification_dataset import ClassificationData
from utils.helpers import read_config, get_device, generate_dirs, train_transforms, test_transforms
from trainers.classification_trainer import ClassificationTrainer


def main():
    parser = argparse.ArgumentParser(description="pytorch based framework for classifcation and segmentation tasks")
    parser.add_argument("--config_path", type=str, required=True, help="Path of the config file")
    args = parser.parse_args()

    configs = read_config(args.config_path)
    device = get_device(configs)

    generate_dirs()

    if(configs['task_type'] == '0'):
        train_data = ClassificationData(train_dir, transform=train_transforms)
        test_data = ClassificationData(test_dir, transform=train_transforms)

        train_dataloader = DataLoader(train_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = True)
        test_dataloader = DataLoader(test_data, batch_size = configs['batch_size'], num_workers=configs['num_workers'], shuffle = False)

        model = model()

        criterion = nn.Softmax()

        optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'])

        trainer = ClassificationTrainer(model, train_dataloader, test_dataloader, optimizer , criterion, device, configs, logger=None)

        results = trainer.train()