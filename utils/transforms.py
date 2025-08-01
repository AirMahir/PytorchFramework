import cv2
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", category=UserWarning, module='albumentations')

TARGET_SIZE_CLASSIFICATION = (384, 384) 

# Training transforms for Classification
train_transforms_classification = A.Compose([
    A.Resize(height=TARGET_SIZE_CLASSIFICATION[0], width=TARGET_SIZE_CLASSIFICATION[1]),
    A.GaussNoise(p=0.2),
    A.PixelDropout(p=0.2),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Transpose(p=1.0),
        A.Rotate(limit=30, p=1.0),
    ], p=0.7),
    A.OneOf([
        A.HueSaturationValue(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.ColorJitter(p=1.0),
        A.CLAHE(p=1.0),
        A.ChannelShuffle(p=1.0),
        A.ChannelDropout(p=1.0),
        A.RGBShift(p=1.0),
        A.RandomToneCurve(p=1.0),
        A.ToGray(p=1.0),
        A.Equalize(p=1.0),
    ], p=0.7),
    A.Normalize(mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Validation transforms for classification
val_transforms_classification = A.Compose([
    A.Resize(height=TARGET_SIZE_CLASSIFICATION[0], width=TARGET_SIZE_CLASSIFICATION[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

TARGET_SIZE_SEGMENTATION = (384, 384) 

# Training transforms for Segmentation
train_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE_SEGMENTATION[0], width=TARGET_SIZE_SEGMENTATION[1]),
    A.GaussNoise(p=0.2),
    A.PixelDropout(p=0.2),
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
        A.Transpose(p=1.0),
        A.Rotate(limit=30, p=1.0),
    ], p=0.7),
    A.OneOf([
        A.HueSaturationValue(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.ColorJitter(p=1.0),
        A.CLAHE(p=1.0),
        A.ChannelShuffle(p=1.0),
        A.ChannelDropout(p=1.0),
        A.RGBShift(p=1.0),
        A.RandomToneCurve(p=1.0),
        A.ToGray(p=1.0),
        A.Equalize(p=1.0),
    ], p=0.7),
    A.Normalize(mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# Validation transforms for Segmentation
val_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE_SEGMENTATION[0], width=TARGET_SIZE_SEGMENTATION[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])