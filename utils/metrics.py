import torch 
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def one_hot_encode(tensors, num_classes):
    """
    Convert [B, H, W] ground truth to one-hot: [B, C, H, W]
    """
    tensors = tensors.long()
    return F.one_hot(tensors, num_classes).permute(0, 3, 1, 2).float()

def calculate_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=1)  # [B, H, W]
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return correct / total

def calculate_iou_score(outputs, targets, num_classes = 3, eps=1e-6):
    preds = torch.argmax(outputs, dim=1) # [B, H, W]
    preds_one_hot = one_hot_encode(preds, num_classes) #[B, C, H, W]
    targets_one_hot = one_hot_encode(targets, num_classes)

    # logger.debug(f"The shape of one-hot encoded vector is : {preds_one_hot.shape}")

    # intersection and union are tensors of dimension = num_classes (C)
    intersection = (preds_one_hot * targets_one_hot).sum(dim=(0, 2, 3))
    union = preds_one_hot.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def calculate_dice_coefficient(outputs, targets, num_classes = 3):
    preds = torch.argmax(outputs, dim=1) # [B, H, W]
    targets = targets.squeeze(1)
    preds_one_hot = one_hot_encode(preds, num_classes)
    targets_one_hot = one_hot_encode(targets, num_classes)

    # logger.debug(f"The shape of one-hot encoded vector is : {targets_one_hot.shape}")

    intersection = (preds_one_hot * targets_one_hot).sum(dim=(0, 2, 3))

    dice = (2 * intersection) / (preds_one_hot.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3)))
    return dice.mean()