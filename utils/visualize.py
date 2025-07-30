import os
import torch
import matplotlib.pyplot as plt

def plot_loss_curves(results, configs):
    """
    Plots the training and validation loss over epochs.

    Args:
        results (dict): Dictionary with 'train_loss' and 'val_loss' lists.
        configs (dict): Configuration dictionary containing 'output_dir' and 'task_name'.
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
    plt.savefig(os.path.join(configs["output_dir"], f'{configs["task_name"]}_loss_curves.png'))
    plt.close() # Close the plot to free memory

def plot_metric_curves(results, configs):
    """
    Plots the training and validation metrics over epochs, adapting to task type.

    Args:
        results (dict): Dictionary containing relevant metric lists.
        configs (dict): Configuration dictionary containing 'output_dir' and 'task_type'.
    """
    if configs['task_type'] == '0':  # Classification
        metrics = ["train_acc", "val_acc"]
        titles = {
            "train_acc": "Train Accuracy", "val_acc": "Val Accuracy"
        }
        subplot_rows = 1
        subplot_cols = 2
        figsize = (15, 5)  # Adjusted figsize for classification
    else:  # Segmentation
        metrics = ["train_acc", "val_acc", "train_iou", "val_iou", "train_dice", "val_dice"]
        titles = {
            "train_acc": "Train Accuracy", "val_acc": "Val Accuracy",
            "train_iou": "Train IoU", "val_iou": "Val IoU",
            "train_dice": "Train Dice", "val_dice": "Val Dice"
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
    plt.savefig(os.path.join(configs["output_dir"], f'{configs["task_name"]}_metric_curves.png'))
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

        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask.astype('uint8'), cmap='jet', vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'batch_{idx}_images.png'))
    plt.close()

def display_segmentation_prediction(images, masks, preds, epoch, configs, class_map=None):
    """
    Saves predictions, masks, and input images for a full batch.
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

        axs[0].imshow(images[idx])
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(masks[idx].astype('uint8'), cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
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
        img = images[i].permute(1, 2, 0).numpy()
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

    images = images.permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
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
