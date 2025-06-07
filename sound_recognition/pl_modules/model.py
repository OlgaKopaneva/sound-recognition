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

        c, h, w = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(c, 48, kernel_size=11, stride=(2, 3), padding=5),
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
            nn.Linear(self._get_flattened_size(c, h, w), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _get_flattened_size(self, c, h, w):
        x = torch.zeros(1, c, h, w)
        x = self.features(x)
        return x.numel()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
