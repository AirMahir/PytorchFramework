import albumentations as transforms
from albumentation.pytorch import ToTensorV2


segmentation_transform =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.CenterCrop((100, 100)),
        transforms.RandomCrop((80, 80)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])