import torch
import torch.nn as nn
from tqdm.auto import tqdm


class SegmentationTrainer(nn.Module):

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config, logger=None):
        super().__init__()

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = logger

        self.scheduler = None
        if "use_scheduler" in config and config["use_scheduler"]:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config["learning_rate"],
                steps_per_epoch=len(train_loader),
                epochs=config["num_epochs"]
            )

    def _train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        correct = 0
        total_pixels = 0

        for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)

            preds = self.model(images)
            loss = self.criterion(preds, masks)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            pred_classes = torch.argmax(preds, dim=1)
            correct += (pred_classes == masks).sum().item()
            total_pixels += torch.numel(masks)

        avg_loss = total_loss / len(self.train_loader)
        acc = correct / total_pixels

        return acc, avg_loss

    def _val_one_epoch(self, epoch):
        self.model.eval()

        total_loss = 0
        correct = 0
        total_pixels = 0

        with torch.no_grad():
            
            for images, masks in tqdm(self.val_loader, desc=f"Epoch {epoch+1} Val", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()

                pred_classes = torch.argmax(preds, dim=1)
                correct += (pred_classes == masks).sum().item()
                total_pixels += torch.numel(masks)

        avg_loss = total_loss / len(self.val_loader)
        acc = correct / total_pixels

        return acc, avg_loss

    def train(self):
        num_epochs = self.config["num_epochs"]
        results = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in tqdm(range(num_epochs), desc="Training Progress"):
            train_acc, train_loss = self._train_one_epoch(epoch)
            val_acc, val_loss = self._val_one_epoch(epoch)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

        return results
