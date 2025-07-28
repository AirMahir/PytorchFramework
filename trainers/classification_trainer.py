import torch
import logging
import torch.nn as nn
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class ClassificationTrainer(nn.Module):

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config, logger=None):

        super().__init__()

        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.device = device
        self.logger = logger

    
    def _train_one_epoch(self, epoch):

        self.model.train()

        train_loss = 0
        train_correct_predictions = 0

        for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train"):

            # logger.debug("The shape here is : ", images.shape)
            
            images, targets = images.to(self.device), targets.to(self.device)
            targets_pred = self.model(images)

            loss = self.criterion(targets_pred, targets)
            train_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            target_pred_class = torch.argmax(targets_pred, dim = 1)
            train_correct_predictions += (target_pred_class == targets).sum().item()

        avg_train_loss = train_loss /  len(self.train_loader)
        avg_train_acc = train_correct_predictions /  len(self.train_loader)

        return avg_train_acc, avg_train_loss

    def _val_one_epoch(self, epoch):

        self.model.eval()

        val_loss = 0
        val_correct_predictions = 0

        with torch.no_grad():

            for (images, targets) in tqdm(self.val_loader, desc = f"Epoch: {epoch+1} Val"):
                
                images, targets = images.to(self.device), targets.to(self.device)

                # 1. Forward pass
                test_preds = self.model(images)

                # 2. Calculate and accumulate loss
                loss = self.criterion(test_preds, targets)
                val_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = torch.argmax(test_preds, dim=1)
                val_correct_predictions += (test_pred_labels == targets).sum().item()
            
        avg_val_loss = val_loss /  len(self.val_loader)
        avg_val_acc = val_correct_predictions /  len(self.val_loader)

        return avg_val_acc, avg_val_loss
    

    def train(self):
        num_epochs = self.config["num_epochs"]

        results = {"train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

        for epoch in tqdm(range(num_epochs)):

            train_acc, train_loss = self._train_one_epoch(epoch)
            val_acc, val_loss = self._val_one_epoch(epoch)

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )

            # 5. Update results dictionary
            # Ensure all data is moved to CPU and converted to float for storage
            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["val_loss"].append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
            results["val_acc"].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)

        return results




