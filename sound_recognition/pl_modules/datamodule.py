from pathlib import Path

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
        self.batch_size = config.data_loading.batch_size
        self.num_workers = config.data_loading.num_workers
        self.conf = auto_complete_conf(self.config.data_loading)

    def setup(self, stage=None):
        df_train = pd.read_csv(self.config.data_loading.meta_path)
        df_test = pd.read_csv(self.config.data_loading.test_meta_path)
        df_train.reset_index(drop=True, inplace=True)

        label2int = {l: i for i, l in enumerate(sorted(df_train.label.unique()))}
        self.num_classes = len(label2int)
        df_train["label_idx"] = df_train.label.map(label2int)

        # Train data

        X_train, idx_train = convert_X(df_train, self.conf, Path(self.config.data_dir))
        y_train = convert_y_train(idx_train, df_train.label_idx.to_numpy())
        self.train_dataset = TensorDataset(
            torch.tensor(X_train).permute(0, 3, 1, 2), torch.tensor(y_train)
        )

        # Test data
        df_test.reset_index(drop=True, inplace=True)
        self.test_fnames = df_test.fname.values
        X_test, idx_test = convert_X(
            df_test, self.conf, Path(self.config.test_data_dir)
        )
        self.test_dataset = TensorDataset(torch.tensor(X_test).permute(0, 3, 1, 2))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
