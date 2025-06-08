import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetLightning(pl.LightningModule):
    """Module for training and evaluation models
    for audio tagging task
    """

    def __init__(self, input_shape, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.validation_preds = []

        input, hide, output = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(input, 48, kernel_size=11, stride=(2, 3), padding=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=5, stride=(2, 3), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 192, kernel_size=3, stride=(1, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(128),
        )

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self._get_flattened_size(input, hide, output), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _get_flattened_size(self, input, hide, output):
        data = torch.zeros(1, input, hide, output)
        data = self.features(data)
        return data.numel()

    def forward(self, data):
        data = self.features(data)
        data = self.flatten(data)
        return self.classifier(data)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def map_k(self, y_true, y_pred, k=3):
        correct = 0
        for true, pred in zip(y_true, y_pred):
            top_k = np.argsort(pred)[::-1][:k]
            if true in top_k:
                correct += 1 / (np.where(top_k == true)[0][0] + 1)
        return correct / len(y_true)

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        labels = labels.to(torch.long)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean()
        map3 = self.map_k(labels, logits)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_map3", map3, prog_bar=True)
        self.validation_preds.append(logits.detach())
        return {"val_loss": loss, "val_acc": acc, "val_map3": map3, "preds": logits}

    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_map3 = torch.stack([x["val_map3"] for x in outputs]).mean()
        self.log("val_loss_epoch", avg_loss)
        self.log("val_acc_epoch", avg_acc)
        self.log("val_map3_epoch", avg_map3)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
