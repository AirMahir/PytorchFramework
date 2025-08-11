import os
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from collections import Counter
import seaborn as sn

def plot_roc_curve(y_true, y_score, config, epoch=None):
    
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    filename = f"roc_curve_epoch_{epoch}.png" if epoch is not None else "roc_curve.png"
    path = os.path.join(config["output_dir"], filename)
    plt.savefig(path)
    plt.close()

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
        subplot_cols = 3
        figsize = (15, 12)  # Adjusted figsize for classification
    else:  # Segmentation
        metrics = ["train_acc", "val_acc", "train_iou", "val_iou", "train_dice", "val_dice", "lr"]
        titles = {
            "train_acc": "Train Accuracy", "val_acc": "Val Accuracy",
            "train_iou": "Train IoU", "val_iou": "Val IoU",
            "train_dice": "Train Dice", "val_dice": "Val Dice",
            "lr": "Learning Rate"
        }
        subplot_rows = 4
        subplot_cols = 2
        figsize = (15, 12)

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
        image = (img - img.min()) / (img.max() - img.min() + 1e-5)
        mask = masks[i].numpy()

        axs[i, 0].imshow(image)
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

        image = (images[idx] - images[idx].min()) / (images[idx].max() - images[idx].min() + 1e-5)
        axs[0].imshow(image)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(masks[idx] , cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[1].set_title("Ground Truth")
        axs[1].axis("off")

        axs[2].imshow(preds[idx], cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[2].set_title("Prediction")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(configs["output_dir"], f"epoch_{epoch}_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

def display_classification_batch(images, targets, idx, output_dir, class_map=None, n=4 ):
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
    targets = targets[:n]

    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3))

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        image = (img  - img .min()) / (img .max() - img .min() + 1e-5)
        target = targets[i]

        axs[i].imshow(image)
        axs[i].set_title(target)
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'dataset_sample_batch_{idx}_images.png'))
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

        image = (images[idx]  - images[idx].min()) / (images[idx].max() - images[idx].min() + 1e-5)
        axs.imshow(image)
        axs.set_title(f"Target: {targets[idx]}, Pred: {preds[idx]}")
        axs.axis("off")

        plt.tight_layout()
        save_path = os.path.join(configs["output_dir"], f"epoch_{epoch}_sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

def plot_tsne(features: np.ndarray, labels: np.ndarray, config, epoch = 0): 
    
    print("Performing t-SNE on feature embeddings...") 
    # 1. Apply t-SNE to reduce dimensionality
    # n_components=2 for a 2D plot
    # perplexity is an important parameter; you might need to tune it (e.g., 5, 30, 50)
    # n_iter is the number of iterations for the optimization
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(features)
    # 2. Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    # Find unique labels and assign a color to each
    unique_labels = np.unique(labels)
    # A colormap is used to get a distinct color for each class
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        # Find the indices corresponding to the current label
        indices = labels == label    
        # Use plt.scatter to plot individual points for each class.
        ax.scatter(tsne_results[indices, 0], tsne_results[indices, 1],
                   label=f'Class {label}', color=colors(i), s=10)

    ax.set_title(f't-SNE of Feature Embeddings - Epoch')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(loc='best', fontsize='small')
    ax.grid(True)
    
    save_path = os.path.join(config["output_dir"], f"tsne_{epoch}.png")
    plt.savefig(save_path)
    plt.close()

def display_confusion_matrix(targets, preds, class_names, epoch, configs):
    """
    Plots a row-normalized confusion matrix for better interpretability.
    Rows = Ground Truth, Columns = Predictions
    """
    cm = confusion_matrix(targets, preds)
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)  # Normalize by GT count

    # Handle division by zero in case a class has 0 samples
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(15, 12))
    sn.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    # Explicit axis labels
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("Ground Truth Label", fontsize=12)

    plt.title(f"Confusion Matrix", fontsize=14)

    save_path = os.path.join(configs["output_dir"], f"inference_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(labels, class_names, save_dir, filename="class_distribution.png"):
    """
    Plots and saves a histogram of the number of samples per class.

    Args:
        labels (list): A list of integer labels for the dataset.
        class_names (list): A list of string class names.
        save_dir (str): The directory to save the plot.
        filename (str): The name of the saved file.
    """
    # Count the number of samples for each class
    label_counts = Counter(labels)
    sorted_labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in sorted_labels]
    
    # Map integer labels to class names for the plot
    sorted_class_names = [class_names[label] for label in sorted_labels]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    sn.barplot(x=sorted_class_names, y=counts, palette="viridis")
    
    # Add counts to the top of the bars
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom', fontsize=12)

    plt.title("Number of Samples per Class", fontsize=16)
    plt.xlabel("Class", fontsize=14)
    plt.ylabel("Number of Samples", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Ensure the save directory exists and save the plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Class distribution plot saved to: {save_path}")

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
