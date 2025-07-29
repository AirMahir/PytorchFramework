import os
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(results):
    plt.figure(figsize=(10, 5))
    plt.plot(results["train_loss"], label="Train Loss")
    plt.plot(results["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metric_curves(results, configs):
    metrics = ["train_acc", "val_acc", "train_iou", "val_iou", "train_dice", "val_dice"]
    titles = {
        "train_acc": "Train Accuracy", "val_acc": "Val Accuracy",
        "train_iou": "Train IoU", "val_iou": "Val IoU",
        "train_dice": "Train Dice", "val_dice": "Val Dice"
    }

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(3, 2, i+1)
        plt.plot(results[metric], label=metric)
        plt.title(titles[metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric.split('_')[-1].capitalize())
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'{configs["task_name"]}_metric_curves.png'))
    plt.show()


def display_segmentation_batch(images, masks, configs, class_map=None, n=4):
    images = images[:n].cpu()
    masks = masks[:n].cpu()

    fig, axs = plt.subplots(n, 2, figsize=(6, 3 * n))

    for i in range(n):
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()

        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mask, cmap='jet', vmin=0, vmax=(len(class_map)-1 if class_map else None))
        axs[i, 1].set_title("Mask")
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], 'segmentation_batch.png'))
    plt.show()

def display_segmentation_prediction(image, mask, pred, count, configs, class_map=None):
    """
    Inputs: image [C, H, W], mask [H, W], pred [H, W] — all torch tensors
    """
    image = image.cpu().permute(1, 2, 0).numpy()
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred, cmap="jet", vmin=0, vmax=(len(class_map)-1 if class_map else None))
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(configs["output_dir"], f'segmentation_prediction_{count}.png'))
    plt.show()