import os
import torch
import numpy as np
import logging
from PIL import Image
from typing import Dict, Tuple, List
from torch.utils.data import Dataset

logger = logging.getLogger(__name__).setLevel(logging.WARNING)

class SegmentationData(Dataset):

    def __init__(self, data_dir:str, transform = None) -> None:


        self.image_dir = os.path.join(data_dir, "Tiles")
        # print(f"The image_dir {self.image_dir}")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_lists = os.listdir(self.image_dir)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_lists)
    
    def __repr__(self):
        return f"SegmentationData(num_samples={len(self)}, image_dir={self.image_dir})"
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.image_lists[index])
        mask_path = os.path.join(self.mask_dir, self.image_lists[index])

        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")
        
        # logger.debug("shape of mask :", np.asarray(mask).shape)

        if self.transform:
            augmented = self.transform(image = np.array(image), mask = np.array(mask).astype(np.float32))
            # print(f"Size of image: {augmented['image'].shape}")
            # print(f"Size of mask: {augmented['mask'].shape}")

            # Albumentations returns masks as np.uint8
            # return augmented["image"], augmented["mask"]
            return augmented["image"].float(), augmented["mask"].float() // 51

        else:
            return image, mask

        