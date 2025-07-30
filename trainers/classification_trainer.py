import os
import torch
import logging
import torch.nn as nn
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.visualize import display_classification_batch, display_classification_prediction, plot_loss_curves, plot_metric_curves


logger = logging.getLogger(__name__)

class ClassificationTrainer(nn.Module):

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, device, config, logger=None):

        super().__init__()

        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.logger = logger 
        self.scaler = GradScaler() # Initialize GradScaler for mixed precision

    def _train_one_epoch(self, epoch):

        self.model.train()

        train_loss = 0
        train_correct_predictions = 0

        for batch_idx, (images, targets) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train")):

            images, targets = images.to(self.device), targets.to(self.device).long()
            
            # Autocast for automatic mixed precision
            with autocast():
                targets_pred = self.model(images)
                loss = self.criterion(targets_pred, targets)
            
            train_loss += loss.item()

            self.optimizer.zero_grad()
            
            # Scale the loss and call backward()
            self.scaler.scale(loss).backward()
            
            # Unscale gradients and update optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()
    
            target_pred_class = torch.argmax(targets_pred, dim = 1)
            train_correct_predictions += (target_pred_class == targets).sum().item()

        avg_train_loss = train_loss / len(self.train_loader)
        avg_train_acc = train_correct_predictions / len(self.train_loader.dataset)
        
        self.logger.info(f"Epoch {epoch+1} Train - Avg Loss: {avg_train_loss:.4f}, Avg Acc: {avg_train_acc*100:.2f}%")

        return avg_train_acc, avg_train_loss
    
    def _val_one_epoch(self, epoch):

        self.model.eval()

        val_loss = 0
        val_correct_predictions = 0

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val", leave=False)):

                if i == 0:
                    display_classification_batch(images, targets, self.config)
                    
                images, targets = images.to(self.device), targets.to(self.device).long()

                # Autocast for automatic mixed precision during validation (no scaler needed)
                with autocast():
                    test_preds = self.model(images)

                if epoch % 5 == 0:
                    display_classification_prediction(images, targets, test_preds, epoch, self.config)

                loss = self.criterion(test_preds, targets)
                val_loss += loss.item()
                
                test_pred_labels = torch.argmax(test_preds, dim=1)
                val_correct_predictions += (test_pred_labels == targets).sum().item()
            
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_acc = val_correct_predictions / len(self.val_loader.dataset)

        self.logger.info(f"Epoch {epoch+1} Val - Avg Loss: {avg_val_loss:.4f}, Avg Acc: {avg_val_acc*100:.2f}%")

        return avg_val_acc, avg_val_loss
    
    def train(self):
        num_epochs = self.config["num_epochs"]

        results = {"train_loss": [],
                   "train_acc": [],
                   "val_loss": [],
                   "val_acc": []
                }
        
        best_acc = 0

        self.logger.info("Starting training process...")

        for epoch in tqdm(range(num_epochs)):

            train_acc, train_loss = self._train_one_epoch(epoch)
            val_acc, val_loss = self._val_one_epoch(epoch)

            print(
                f"Epoch: {epoch+1:02d} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc*100:.2f}% | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc*100:.2f}%"
            )
            self.logger.info(f"Epoch Summary: Epoch: {epoch+1:02d} | train_loss: {train_loss:.4f} | train_acc: {train_acc*100:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}%")

            if(val_acc > best_acc):
                self.logger.info(f"Validation accuracy improved from {best_acc*100:.2f}% to {val_acc*100:.2f}%. Saving best model...")
                best_acc = val_acc
                model_path = os.path.join(self.config["output_dir"], f"best_checkpoint.pth")

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

            # 5. Update results dictionary
            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["val_loss"].append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
            results["val_acc"].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)
        
        self.logger.info("Training process finished.")
        plot_loss_curves(results, self.config)
        plot_metric_curves(results, self.config)
        self.logger.info(f"Loss and metric curves saved to {self.config["output_dir"]}")
        return results