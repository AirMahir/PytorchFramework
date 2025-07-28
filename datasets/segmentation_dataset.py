import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
from torch.utils.data import Dataset

class SegmentationData(Dataset):

    def __init__(self, data_dir:str, transform = None) -> None:

        self.image_dir = os.path.join(data_dir, "Tiles")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_lists = os.listdir(self.image_dir)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_lists)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.image_lists[index])
        mask_path = os.path.join(self.mask_dir, self.image_lists[index])

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transform:
            augmented = self.transform(image = np.array(image), mask = np.array(mask))
            return augmented["image"], augmented["mask"]
        else:
            return image, mask

        