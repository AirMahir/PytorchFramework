import os
import cv2
import torch
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None, is_inference: bool=False, logger=None):
        self.image_dir = image_dir
        self.transform = transform
        self.logger = logger
        self.is_inference = is_inference

        if not is_inference:
            self.data = pd.read_csv(csv_path)
            assert 'Image-Name' in self.data.columns and 'Label' in self.data.columns
            assert self.data['Label'].isna().sum() == 0
            self.paths = self.data['Image-Name'].tolist()[:1000]

            if self.logger:
                self.logger.info(f"Loaded {len(self.paths)} labeled images from {image_dir}")
                self.logger.info(f"Label distribution: {self.data['Label'].value_counts().to_dict()}")
        else:
            self.paths = list(pathlib.Path(image_dir).glob("*.png"))
            if self.logger:
                self.logger.info(f"Loaded {len(self.paths)} images for inference from {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.is_inference:
            image_path = self.paths[index]
        else:
            img_name = str(self.paths[index])
            image_path = os.path.join(self.image_dir, img_name)
            label = self.data.iloc[index]['Label']

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.is_inference:
            return image
        else:
            # print(image.shape, label)
            return image, torch.tensor(label, dtype=torch.long)
