import os
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
        image = Image.open(img_path)

        if self.is_inference:
            if self.transform:
                augmented = self.transform(image = np.array(image))
                return augmented["image"].float(), img_name
            else:
                return image, img_name
        else:
            mask_path = os.path.join(self.mask_dir, img_name)
            mask = Image.open(mask_path).convert("L")

            mask_np = np.array(mask)
            binary_mask = (mask_np > 153).astype(np.float32)

            if self.transform:
                augmented = self.transform(image = np.array(image), mask = np.array(binary_mask).astype(np.float32))
                return augmented["image"].float(), augmented["mask"].float() 
            else:
                return torch.tensor(np.array(image)), torch.tensor(np.array(mask).astype(np.float64))

        