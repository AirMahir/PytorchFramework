import torch
import pathlib
from PIL import Image
import numpy as np
from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, Dataset
from utils.helpers import find_classes


class ClassificationDataset(Dataset):

    def __init__(self, image_dir:str, transform = None, is_inference: bool = False, logger = None) -> None:

        self.image_dir = image_dir
        if is_inference:
            self.paths = list(pathlib.Path(image_dir).glob("*.jpg"))
        else:
            self.paths = list(pathlib.Path(image_dir).glob("*/*.jpg"))
            self.classes, self.class_to_idx = find_classes(image_dir)

        self.transform = transform
        self.logger = logger
        self.is_inference = is_inference

        print(f"Loaded {len(self.paths)} images from {image_dir}")

    def __repr__(self):
        self.logger.info(f"ClassificationDataset(num_samples={len(self)}, image_dir={self.image_dir})")
        return f"ClassificationDataset(num_samples={len(self)}, image_dir={self.image_dir})"
    
    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path).convert("RGB") 
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name 
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            transformed_img = self.transform(image=np.array(img))["image"]
        else:
            transformed_img = img

        if self.is_inference:
            return transformed_img, str(self.paths[index].name)
        else:
            class_name  = self.paths[index].parent.name 
            class_idx = self.class_to_idx[class_name]
            return transformed_img, class_idx