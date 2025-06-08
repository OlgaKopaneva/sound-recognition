from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pl_modules.preprocessing import auto_complete_conf, convert_X, convert_y_train
from torch.utils.data import DataLoader, TensorDataset


class SoundDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.batch_size = config.training.batch_size
        self.conf = auto_complete_conf(self.config.preprocessing)

    def setup(self, stage=None):
        df_train = pd.read_csv(self.config.data_loading.meta_path)
        df_test = pd.read_csv(self.config.data_loading.test_meta_path)

        labels = df_train.label.unique()
        label2int = {l: i for i, l in enumerate(labels)}
        self.num_classes = len(labels)

        # Plain y_train label
        plain_y_train = np.array([label2int[label] for label in df_train.label])

        # Train data
        data_train, idx_train = convert_X(
            df_train, self.conf, Path(self.config.data_dir)
        )
        label_train = convert_y_train(idx_train, plain_y_train)
        self.train_dataset = TensorDataset(
            torch.tensor(data_train).permute(0, 3, 1, 2), torch.tensor(label_train)
        )

        # Test data
        self.test_fnames = df_test.fname.values
        data_test, idx_test = convert_X(
            df_test, self.conf, Path(self.config.test_data_dir)
        )
        self.test_dataset = TensorDataset(torch.tensor(data_test).permute(0, 3, 1, 2))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
