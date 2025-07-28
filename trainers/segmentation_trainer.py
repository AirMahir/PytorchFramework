import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ..utils.metrics import calculate_accuracy, calculate_dice_coefficient, calculate_iou_score

class SegmentationTrainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, device, config, logger=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger

        self.scheduler = scheduler

    def _train_one_epoch(self, epoch, num_channels):
        self.model.train()

        total_loss = 0
        accuracy = []
        iou_score = []
        dice_coeff = []

        for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)

            preds = self.model(images)
            self.logger.debug(f"During training : preds shape = {preds.shape}")
            self.logger.debug(f"During trainig : masks shape = {masks.shape}")

            loss = self.criterion(preds, masks)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            accuracy.append(calculate_accuracy(preds, masks))
            iou_score.append(calculate_iou_score(preds, masks, num_classes = num_channels))
            dice_coeff.append(calculate_dice_coefficient(preds, masks, num_classes = num_channels))

        return total_loss / len(self.train_loader), accuracy.mean().item(), iou_score.mean().item(), dice_coeff.mean().item()

    def _val_one_epoch(self, epoch, num_channels):
        self.model.eval()

        total_loss = 0
        accuracy = []
        iou_score = []
        dice_coeff = []

        with torch.no_grad():

            for images, masks in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()

                accuracy.append(calculate_accuracy(preds, masks))
                iou_score.append(calculate_iou_score(preds, masks, num_channels))
                dice_coeff.append(calculate_dice_coefficient(preds, masks, num_channels))

        return total_loss / len(self.val_loader), accuracy.mean().item(), iou_score.mean().item(), dice_coeff.mean().item()

    def train(self):
        num_epochs = self.config["num_epochs"]
        results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        num_channels = self.config["num_channels"]
        
        best_acc = 0

        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            train_loss, train_acc, train_iou, train_dice = self._train_one_epoch(epoch, num_channels)
            val_loss, val_acc, val_iou, val_dice = self._val_one_epoch(epoch, num_channels)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train IOU : {train_iou:.4f}, Train DiceCoeff : {train_dice:.4f}| "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val IOU : {val_iou:.4f}, Val DiceCoeff : {val_dice:.4f}"
            )

            if(val_acc > best_acc):
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.config["output_dir"])

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["train_iou"].append(train_iou)
            results["train_dice"].append(train_dice)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["val_iou"].append(val_iou)
            results["val_dice"].append(val_dice)

        return results
