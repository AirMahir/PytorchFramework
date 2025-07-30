import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(results, configs):
    """
    Plots the training and validation loss over epochs.

    Args:
        results (dict): Dictionary with 'train_loss' and 'val_loss' lists.
        configs (dict): Configuration dictionary containing 'output_dir' and 'task_type'.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot instead of showing it
    os.makedirs(configs["output_dir"], exist_ok=True)
    plt.savefig(os.path.join(configs["output_dir"], f'{configs["task_type"]}_loss_curves.png'))
    plt.close() # Close the plot to free memory

def plot_metric_curves(results, configs):
    """
    Plots the training and validation metrics over epochs, adapting to task type.

    Args:
        results (dict): Dictionary containing relevant metric lists.
        configs (dict): Configuration dictionary containing 'output_dir' and 'task_type'.
    """
    if configs['task_type'] == 'classification':  # Classification
        metrics = ["train_acc", "val_acc", "lr"]
        titles = {
            "train_acc": "Train Accuracy", "val_acc": "Val Accuracy", "lr": "Learning Rate"
        }
        subplot_rows = 1
        subplot_cols = 2
        figsize = (15, 5)  # Adjusted figsize for classification
    else:  # Segmentation
        metrics = ["train_acc", "val_acc", "train_iou", "val_iou", "train_dice", "val_dice", "lr"]
        titles = {
            "train_acc": "Train Accuracy", "val_acc": "Val Accuracy",
            "train_iou": "Train IoU", "val_iou": "Val IoU",
            "train_dice": "Train Dice", "val_dice": "Val Dice",
            "lr": "Learning Rate"
        }
        subplot_rows = 3
        subplot_cols = 2
        figsize = (15, 10)

    plt.figure(figsize=figsize)

    for i, metric in enumerate(metrics):
        plt.subplot(subplot_rows, subplot_cols, i + 1)
        plt.plot(results[metric], label=metric)
        plt.title(titles[metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric.split('_')[-1].capitalize())
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'{configs["task_type"]}_metric_curves.png'))
    plt.close() # Close the plot to free memory

def display_segmentation_batch(images, masks, idx, configs, class_map=None, n=4):
    """
    Saves a batch of input images and corresponding segmentation masks.

    Args:
        images (Tensor): Input images tensor of shape [B, C, H, W].
        masks (Tensor): Segmentation masks tensor of shape [B, H, W] or [B, 1, H, W].
        configs (dict): Configuration dictionary with 'output_dir'.
        class_map (dict, optional): Dictionary mapping class indices to labels.
        n (int): Number of samples to display.
    """
    images = images[:n].cpu()
    masks = masks[:n].cpu()

    fig, axs = plt.subplots(n, 2, figsize=(6, 3 * n))

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()

        axs[i, 0].imshow((img * 255).astype(np.uint8))
        axs[i, 0].set_title("Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask.astype('uint8'), cmap='jet', vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'dataset_sample_batch_{idx}_images.png'))
    plt.close()

def display_segmentation_prediction(images, masks, preds, epoch, configs, class_map=None):
    """
    Saves predictions, masks, and input images for ALL samples in the batch.
    This function is called every 5th epoch to visualize model predictions.
    
    Inputs:
        images: Tensor [B, C, H, W]
        masks: Tensor [B, H, W]
        preds: Tensor [B, C, H, W] or [B, H, W] if already argmaxed
    """

    images = images.detach().cpu()
    masks = masks.detach().cpu()
    preds = preds.detach().cpu()

    preds = torch.argmax(preds, dim=1)

    images = images.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
    masks = masks.numpy()
    preds = preds.numpy()

    for idx in range(images.shape[0]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow((images[idx]*255).astype(np.uint8))
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow((masks[idx] * 255).astype('uint8'), cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(preds[idx], cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(configs["output_dir"], f"epoch_{epoch}_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

def display_classification_batch(images, targets, idx, configs, class_map=None, n=4):
    """
    Saves a batch of input classification images with their ground truth labels.

    Args:
        images (Tensor): Input images tensor of shape [B, C, H, W].
        targets (Tensor): Class labels tensor of shape [B].
        configs (dict): Configuration dictionary with 'output_dir'.
        class_map (dict, optional): Dictionary mapping class indices to class names.
        n (int): Number of images to display.
    """
    images = images[:n].cpu()
    targets = targets[:n].cpu()

    fig, axs = plt.subplots(n, 1, figsize=(6, 3 * n))

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy().astype("uint8")
        target = targets[i].numpy()

        axs[i].imshow(img)
        axs[i].set_title(target)
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'batch_{idx}_images.png'))
    plt.close()

def display_classification_prediction(images, targets, preds, epoch, configs, class_map=None):
    """
    Saves predictions, targets, and input images for a full batch.
    Inputs:
        images: Tensor [B, C, H, W]
        targets: Tensor [B, 1]
        preds: Tensor [B, C] or [B, 1] if already argmaxed
    """

    images = images.detach().cpu()
    targets = targets.detach().cpu()
    preds = preds.detach().cpu()

    if preds.ndim == 2:  # logits/scores
        preds = torch.argmax(preds, dim=1)

    images = images.permute(0, 2, 3, 1).numpy().astype("uint8")  # (B, H, W, C)
    targets = targets.numpy()
    preds = preds.numpy()

    for idx in range(images.shape[0]):
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))

        axs.imshow(images[idx])
        axs.set_title(f"Target: {targets[idx]}, Pred: {preds[idx]}")
        axs.axis("off")

        plt.tight_layout()
        save_path = os.path.join(configs["output_dir"], f"epoch_{epoch}_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

def explore_segmentation_dataset(dataloader, configs, num_batches=3, samples_per_batch=4, class_map=None):
    """
    Utility function to explore and visualize segmentation dataset samples.
    This should be called before training to understand the dataset.
    
    Args:
        dataloader: DataLoader containing the dataset
        configs (dict): Configuration dictionary with 'output_dir'
        num_batches (int): Number of batches to visualize
        samples_per_batch (int): Number of samples to show per batch
        class_map (dict, optional): Dictionary mapping class indices to labels
    """
    print(f"Exploring segmentation dataset...")
    print(f"Will visualize {num_batches} batches with {samples_per_batch} samples each")
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"Visualizing batch {batch_idx + 1}/{num_batches}")
        display_segmentation_batch(images, masks, batch_idx, configs, class_map, samples_per_batch)
    
    print(f"Dataset exploration complete. Check {configs['output_dir']} for visualization files.")
