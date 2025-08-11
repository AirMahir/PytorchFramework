import os
import sys
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp import autocast
from torcheval.metrics import MulticlassF1Score
from sklearn.metrics import roc_auc_score
from utils.visualize import plot_roc_curve, display_classification_prediction, plot_loss_curves, plot_metric_curves
from utils.helpers import set_pytorch_optimizations, calculate_time

# Set PyTorch performance optimizations
set_pytorch_optimizations()

class ClassificationTrainer(nn.Module):

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device, config, writer, logger=None, start_epoch = 0):

        super().__init__()

        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler # Initialize GradScaler for mixed precision
        self.device = device
        self.config = config
        self.writer = writer
        self.logger = logger 

        self.start_epoch = start_epoch

    @calculate_time
    def _train_one_epoch(self, epoch):

        self.model.train()

        train_loss = 0
        train_correct_predictions = 0
        all_probs_list = []
        all_targets_list = []
        accum_iter = 4
        self.optimizer.zero_grad()

        f1_metric = MulticlassF1Score(num_classes=self.config["data"]["num_classes"], average=None).to(self.device)

        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", file=sys.stderr)):

            images, targets = images.to(self.device), targets.to(self.device).long()

            # self.optimizer.zero_grad()
        
            # Autocast for automatic mixed precision
            with torch.amp.autocast('cuda', enabled=True):
                targets_pred = self.model(images)
                loss = self.criterion(targets_pred, targets)
                loss = loss / accum_iter
            
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % accum_iter == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            probs = torch.softmax(targets_pred, dim=1)
            train_loss += loss.item() * accum_iter 
            target_pred_class = torch.argmax(targets_pred, dim = 1)
            train_correct_predictions += (target_pred_class == targets).sum().item()
            all_probs_list.append(probs.cpu())
            all_targets_list.append(targets.cpu())

        all_probs = torch.cat(all_probs_list, dim=0)
        all_targets = torch.cat(all_targets_list, dim=0)

        f1_metric.update(all_probs.to(self.device), all_targets.to(self.device))
        per_class_f1 = f1_metric.compute()
        macro_f1 = per_class_f1.mean().item()
        
        for i, score in enumerate(per_class_f1):
            self.writer.add_scalar(f"F1_train/Class_{i}", score.item(), epoch)
        self.writer.add_scalar("F1_train/Macro_F1", macro_f1, epoch)

        avg_train_loss = train_loss / len(self.train_loader)
        avg_train_acc = train_correct_predictions / len(self.train_loader.dataset)

        return avg_train_loss, avg_train_acc
    
    @calculate_time
    def _val_one_epoch(self, epoch):

        self.model.eval()

        val_loss = 0
        val_correct_predictions = 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val", leave=True)):

                images, targets = images.to(self.device), targets.to(self.device).long()

                with autocast(device_type=self.device.type):
                    test_preds = self.model(images)
                    loss = self.criterion(test_preds, targets)

                val_loss += loss.item()
                probs = torch.softmax(test_preds, dim=1)
                all_probs.append(probs.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

                if epoch % self.config["training"]["log_interval"] == 0:
                    display_classification_prediction(images, targets, test_preds, epoch, self.config)

                test_pred_labels = torch.argmax(test_preds, dim=1)
                val_correct_predictions += (test_pred_labels == targets).sum().item()
            
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_acc = val_correct_predictions / len(self.val_loader.dataset)

        all_probs = np.concatenate(all_probs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        auc_score = roc_auc_score(all_targets, all_probs[:, 1])

        if epoch % self.config["training"]["save_interval"] == 0 or epoch == self.config["training"]["epochs"] - 1:
            plot_roc_curve(all_targets, all_probs[:, 1], self.config, epoch)

        return avg_val_loss, avg_val_acc, auc_score
    
    def train(self):
        num_epochs = self.config["training"]["epochs"]

        results = {"train_loss": [],
                   "train_acc": [],
                   "val_loss": [],
                   "val_acc": [],
                   "auc": [],
                   "lr": []
                }
        
        best_loss = 100

        self.logger.info("Starting training process...")

        for epoch in tqdm(range(self.start_epoch, num_epochs)):

            (train_loss, train_acc), _train_time = self._train_one_epoch(epoch)
            (val_loss, val_acc, auc), _val_time = self._val_one_epoch(epoch)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Acc/train", train_acc, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Acc/Val", val_acc, epoch)
            self.writer.add_scalar("AUC/epoch", auc, epoch)
        
            self.scheduler.step(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("learning Rate", current_lr, epoch)

            self.logger.info(
                f"Epoch: {epoch+1:02d} | "
                f"train_loss: {train_loss:.4f} || train_acc: {train_acc * 100:.2f}% | "
                f"val_loss: {val_loss:.4f} || val_acc: {val_acc * 100:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"train_time: {_train_time:.2f}s | val_time: {_val_time:.2f}s || "
                f"AUC-ROC score: {auc:.4f}"
            )

            if(val_loss < best_loss):
                self.logger.info(f"Validation loss decreased from {best_loss*100:.2f}% to {val_loss*100:.2f}%. Saving best model...")
                best_loss = val_loss
                model_path = os.path.join(self.config["output_dir"], f"best_{self.config["task_type"]}_checkpoint.pth")

                checkpoint = {
                    'epoch' : epoch,
                    'model': self.model.state_dict(),  
                    'criterion_state_dict': self.criterion.state_dict(),
                    'optimizer_state_dict' : self.optimizer.state_dict(),
                    'lr_scheduler_state_dict' : self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict() # Save scaler state
                }
                torch.save(checkpoint, model_path)
                self.logger.info(f"Best model saved to {model_path}")

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

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            results["auc"].append(auc.item() if isinstance(auc, torch.Tensor) else auc)
            results["lr"].append(current_lr)

        self.logger.info("Training process finished.")
        plot_loss_curves(results, self.config)
        plot_metric_curves(results, self.config)
        self.logger.info(f"Loss and metric curves saved to {self.config["output_dir"]}")
        return results