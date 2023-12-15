import torch
from tqdm import tqdm
from kernfont import logger


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        experiment,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment = experiment

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_accuracy = 0
            for (sx, dx, kern_value), target in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}"
            ):
                (sx, dx, kern_value), target = (sx, dx, kern_value).to(
                    self.device
                ), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(sx, dx, kern_value)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * (sx, dx, kern_value).size(0)
                train_accuracy += (output.argmax(dim=1) == target).sum().item()
                self.experiment.log_metrics(
                    {"loss": train_loss.item(), "accuracy": train_accuracy.item()},
                    step=epoch + 1,
                )
            train_loss /= len(self.train_loader.dataset)
            train_accuracy /= len(self.train_loader.dataset)
            self.experiment.log_metrics(
                {"loss": train_loss, "accuracy": train_accuracy},
                step=epoch,
            )
            logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
            )
