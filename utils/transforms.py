import albumentations as A

train_transforms_classification = A.Compose([
    A.Resize(64, 64),  # Slightly larger than target
    A.RandomCrop(32, 32),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.Rotate(limit=10, p=0.5),  # Small rotation
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

# Validation transforms - deterministic
val_transforms_classification = A.Compose([
    A.Resize(32, 32),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])


TARGET_SIZE = (1444, 1444)

train_transform_segmentation = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE[0] * 2, p=1.0),
    A.RandomCrop(height=TARGET_SIZE[0], width=TARGET_SIZE[1], p=1.0),
    A.SquareSymmetry(p=1.0),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(std_range=(0.1, 0.2), p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

val_transform_segmentation = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE[0] * 2, p=1.0),
    A.CenterCrop(height=TARGET_SIZE[0] * 2, width=TARGET_SIZE[1] * 2, p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
])

