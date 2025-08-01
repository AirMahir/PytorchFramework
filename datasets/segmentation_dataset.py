import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

    def __init__(self, data_dir:str, transform = None, is_inference: bool = False) -> None:

        self.is_inference = is_inference
        if self.is_inference:
            self.image_dir = data_dir
            self.image_lists = os.listdir(self.image_dir)
        else:
            self.image_dir = os.path.join(data_dir, "Tile")
            self.mask_dir = os.path.join(data_dir, "Mask")
            self.image_lists = os.listdir(self.image_dir)

        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_lists)
    
    def __repr__(self):
        return f"SegmentationDataset(num_samples={len(self)}, data_dir={self.image_dir})"
    
    def __getitem__(self, index: int):
        img_name = self.image_lists[index]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.open(img_path)

        if self.is_inference:
            if self.transform:
                augmented = self.transform(image = np.array(image))
                return augmented["image"].float(), img_name
            else:
                return image, img_name
        else:
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            mask = (mask >= 153).astype(np.uint8)

            if self.transform:
                augmented = self.transform(image = np.array(image), mask = np.array(mask))
                return augmented["image"].float(), augmented["mask"].float() 
            else:
                return torch.tensor(np.array(image)), torch.tensor(np.array(mask))

        