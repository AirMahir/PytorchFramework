import numpy as np
from torch.utils.data import DataLoader
from utils.helpers import read_config, seed_everything, get_device, set_pytorch_optimizations, generate_dirs
from datasets.classification_dataset import ClassificationDataset
from utils.transforms import train_transforms_classification
from utils.visualize import display_classification_batch, plot_class_distribution

# Set PyTorch performance optimizations
set_pytorch_optimizations()

def main():
    # Load configuration
    config_path = "configs/configs_classification.json"
    
    config = read_config(config_path)
    generate_dirs(config)
    seed_everything(config["seed"])
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    train_dataset = ClassificationDataset(
        image_dir=config["data"]["train_dir"],
        csv_path=config['data']["train_csv"],
        transform=train_transforms_classification
    )

    # plot_class_distribution(
    #     labels=train_dataset.labels,
    #     class_names=train_dataset.class_names,
    #     save_dir="Data"
    # )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True
    )

    # Explore the dataset
    print("=== Classification Dataset Exploration ===")
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        if batch_idx >= 10:  # Visualize 3 batches
            break
        display_classification_batch(images, targets, batch_idx, "Data", n=8)

if __name__ == "__main__":
    main() 