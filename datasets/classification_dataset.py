import torch
import pathlib
import logging
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, Dataset
from utils.helpers import find_classes

logger = logging.getLogger(__name__)

class ClassificationData(Dataset):

    def __init__(self, image_dir:str, transform = None) -> None:

        self.paths = list(pathlib.Path(image_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(image_dir)

        print(f"Loaded {len(self.paths)} images from {image_dir}")

    def __repr__(self):
        return f"ClassificationData(num_samples={len(self)}, image_dir={self.image_dir})"
    
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name 
        class_idx = self.class_to_idx[class_name]

        # print(f"Shape for image : {img.size}")

        if self.transform:
            transformed = self.transform(image=np.array(img))  # return data, label (X, y)
            # print(f"Shape for class_idx : {class_idx}")
            # print(f"Shape for image : {transformed["image"].shape}")
            return transformed["image"], class_idx 
        else:
            return img, class_idx  # return data, label (X, y)
