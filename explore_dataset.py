from torch.utils.data import DataLoader
from utils.helpers import read_config, get_device, set_pytorch_optimizations, generate_dirs
from datasets.classification_dataset import ClassificationDataset
from datasets.segmentation_dataset import SegmentationDataset
from utils.transforms import train_transforms_classification, train_transform_segmentation
from utils.visualize import display_classification_batch, display_segmentation_batch

# Set PyTorch performance optimizations
set_pytorch_optimizations()

def main():
    # Load configuration
    task_type = "segmentation"  # or "classification"
    
    if task_type == "segmentation":
        config_path = "configs/configs_segmentation.json"
    else:
        config_path = "configs/configs_classification.json"
    
    config = read_config(config_path)
    generate_dirs(config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    if task_type == "segmentation":
        train_dataset = SegmentationDataset(
            data_dir=config["data"]["train_dir"],
            transform=None
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True
        )
        
        # Explore the dataset
        print("=== Segmentation Dataset Exploration ===")
        for batch_idx, (images, masks) in enumerate(train_loader):
            if batch_idx >= 3:  # Visualize 3 batches
                break
            display_segmentation_batch(images, masks, batch_idx, config, n=4)
        
    else:  # classification
        train_dataset = ClassificationDataset(
            data_dir=config["data"]["train_dir"],
            transform=train_transforms_classification
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            pin_memory=True
        )
        
        # Explore the dataset
        print("=== Classification Dataset Exploration ===")
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= 3:  # Visualize 3 batches
                break
            display_classification_batch(images, targets, batch_idx, config, n=4)

if __name__ == "__main__":
    main() 