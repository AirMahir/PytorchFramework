import os
import torch
from tqdm.auto import tqdm
from torch.amp import autocast
from utils.metrics import calculate_accuracy, calculate_dice_coefficient, calculate_iou_score
from utils.visualize import display_segmentation_batch, display_segmentation_prediction, plot_metric_curves
from utils.helpers import set_pytorch_optimizations, calculate_time

# Set PyTorch performance optimizations
set_pytorch_optimizations()

class SegmentationTrainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device, config, logger=None, start_epoch = 0):
        
        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler 
        self.device = device
        self.config = config
        self.logger = logger 

        self.start_epoch = start_epoch

    @calculate_time
    def _train_one_epoch(self, epoch, num_classes):
        self.model.train()

        total_loss = 0
        accuracy = []
        iou_score = []
        dice_coeff = []

        for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            masks = masks.long()

            # Autocast for automatic mixed precision
            with autocast(device_type=self.device.type):
                preds = self.model(images)
                loss = self.criterion(preds, masks)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            # Scale the loss and call backward()
            self.scaler.scale(loss).backward()
            # Unscale gradients and update optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            accuracy.append(calculate_accuracy(preds, masks))
            iou_score.append(calculate_iou_score(preds, masks, num_classes=num_classes))
            dice_coeff.append(calculate_dice_coefficient(preds, masks, num_classes=num_classes))

        return (
            total_loss / len(self.train_loader),
            torch.tensor(accuracy).mean().item(),
            torch.tensor(iou_score).mean().item(),
            torch.tensor(dice_coeff).mean().item()
        )

    @calculate_time
    def _val_one_epoch(self, epoch, num_classes):
        self.model.eval()

        total_loss = 0
        accuracy = []
        iou_score = []
        dice_coeff = []

        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val", leave=False)):

                # if i == 0:
                #     display_segmentation_batch(images, masks, i, self.config)

                images, masks = images.to(self.device), masks.to(self.device)
                masks = masks.long()
                
                # No GradScaler needed in validation as there's no backward pass
                with autocast(device_type=self.device.type):
                    preds = self.model(images)
                    loss = self.criterion(preds, masks)

                total_loss += loss.item()

                if(epoch % 5 == 0):
                    display_segmentation_prediction(images, masks, preds, epoch, self.config)

                accuracy.append(calculate_accuracy(preds, masks))
                iou_score.append(calculate_iou_score(preds, masks, num_classes))
                dice_coeff.append(calculate_dice_coefficient(preds, masks, num_classes))

            return (
                total_loss / len(self.val_loader),
                torch.tensor(accuracy).mean().item(),
                torch.tensor(iou_score).mean().item(),
                torch.tensor(dice_coeff).mean().item()
            )
        
    def train(self):
        num_epochs = self.config["training"]["epochs"]
        results = {
            "train_loss": [],
            "train_acc": [],
            "train_iou": [],
            "train_dice": [],
            "val_loss": [],
            "val_acc": [],
            "val_iou": [],
            "val_dice": [],
            "lr": []
        }

        num_classes = self.config["data"]["num_classes"]
        
        best_loss = 100

        for epoch in tqdm(range(self.start_epoch, num_epochs), desc="Training Progress"):
            (train_loss, train_acc, train_iou, train_dice), _train_time = self._train_one_epoch(epoch, num_classes)
            (val_loss, val_acc, val_iou, val_dice), _val_time = self._val_one_epoch(epoch, num_classes)

            self.scheduler.step(epoch)

            current_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"Epoch {epoch+1:02d} | "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | IoU: {train_iou*100:.2f}% | Dice: {train_dice*100:.2f}% || "
                f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | IoU: {val_iou*100:.2f}% | Dice: {val_dice*100:.2f}% ||"
                f" LR: {current_lr:.6f} | train_time: {_train_time:.2f}s | val_time: {_val_time:.2f}s"
            )

            if(val_loss < best_loss):
                best_loss = val_loss
                model_path = os.path.join(self.config["checkpoints_dir"], f"best_{self.config["task_type"]}_checkpoint.pth")

                checkpoint = {
                    'epoch' : epoch,
                    'model': self.model.state_dict(),  
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'lr_scheduler_state_dict' : self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() 
                } 
                torch.save(checkpoint, model_path)

            if(epoch % 5 == 0):
                model_path = os.path.join(self.config["checkpoints_dir"], f"{self.config["task_type"]}__{epoch}_checkpoint.pth")
                checkpoint = {
                    'epoch' : epoch,
                    'model': self.model.state_dict(),  
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'lr_scheduler_state_dict' : self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() 
                } 
                torch.save(checkpoint, model_path)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["train_iou"].append(train_iou)
            results["train_dice"].append(train_dice)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["val_iou"].append(val_iou)
            results["val_dice"].append(val_dice)
            results["lr"].append(current_lr)

        plot_metric_curves(results, self.config)
        return results