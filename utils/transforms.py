import albumentations as A

train_transforms_classification = A.Compose([
    A.Resize(256, 256),  # Slightly larger than target
    A.RandomCrop(128, 128),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

# Validation transforms - deterministic
val_transforms_classification = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])


TARGET_SIZE = (512, 512) 

train_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),
    A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

val_transform_segmentation = A.Compose([
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])
