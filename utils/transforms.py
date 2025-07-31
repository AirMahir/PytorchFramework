import cv2
import warnings
import albumentations as A

warnings.filterwarnings("ignore", category=UserWarning, module='albumentations')


# Training transforms for Classification
train_transforms_classification = A.Compose([
    A.Resize(256, 256),  # Slightly larger than target
    A.RandomCrop(128, 128),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

# Validation transforms for classification
val_transforms_classification = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])


TARGET_SIZE = (384, 284) 

# Training transforms for Segmentation
# train_transform_segmentation = A.Compose([
#     A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),

#     A.HorizontalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
#     A.ElasticTransform(alpha=1, sigma=50, p=0.2, border_mode=cv2.BORDER_REFLECT), 
#     A.RandomBrightnessContrast(p=0.3),
#     A.GaussNoise(p=0.2),
#     A.Blur(blur_limit=3, p=0.1),
#     A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
#     A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means/stds
#                 std=(0.229, 0.224, 0.225)),
#     A.ToTensorV2(),
# ])

train_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    # A.ElasticTransform(alpha=1, sigma=50, p=0.2, border_mode=cv2.BORDER_REFLECT), # Keep if you find it essential
    A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means/stds (adjust if medical images have different distribution)
                std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

# Validation transforms for Segmentation
val_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])
