import torch
import torch.nn as nn
from tqdm.auto import tqdm


class ClassificationTrainer(nn.module):

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config, logger=None):

        self.model = model 
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.device = self.device
        self.logger = logger

    
    def _train_one_epoch(self, epoch):

        self.model.train()

        train_loss = 0
        train_acc = 0

        for batch, (images, targets) in enumerate(self.train_loader):
            
            images, targets = images.to(self.device), targets.to(self.device)

            targets_pred = self.model(images)

            loss = self.criterion(targets_pred, targets)
            train_loss += loss.item()

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            target_pred_class = torch.argmax(torch.softmax(targets_pred, dim = 1), dim = 1)
            train_acc += (target_pred_class == targets).sum().item()/len(targets_pred)

        return train_acc, train_loss

    def _val_one_epoch(self, epoch):

        self.model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():

            for batch, (images, targets) in enumerate(self.val_loader):
                
                images, targets = images.to(self.device), targets.to(self.device)

                # 1. Forward pass
                test_preds = self.model(images)

                # 2. Calculate and accumulate loss
                loss = self.criterion(test_preds, targets)
                val_loss += loss.item()
                
                # Calculate and accumulate accuracy
                test_pred_labels = test_preds.argmax(dim=1)
                val_acc += ((test_pred_labels == targets).sum().item()/len(test_pred_labels))
            

        return val_acc, val_loss

    def train(self):
        num_epochs = self.config["num_epochs"]

        results = {"train_loss": [],
                "train_acc": [],
                "val_loss": [],
                "val_acc": []
            }

        for epoch in tqdm(range(num_epochs)):

            if(epoch % self.config['number_val']):
                val_acc, val_loss = self._val_one_epoch(self, epoch)
                print(
                    f"Epoch: {epoch+1} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"val_acc: {val_acc:.4f}"
                )

            train_acc, train_loss = self._train_one_epoch(self, epoch)

            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f}"
            )

            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["val_loss"].append(val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss)
            results["val_acc"].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)


        return results




