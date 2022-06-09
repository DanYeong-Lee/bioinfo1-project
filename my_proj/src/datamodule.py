import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl


class OneHotDataset(Dataset):
    def __init__(
        self, 
        df
    ):
        self.records = df.to_records()
        self.base2vec = {
            "A": [1., 0., 0., 0.],
            "T": [0., 1., 0., 0.],
            "C": [0., 0., 1., 0.],
            "G": [0., 0., 0., 1.],
        }
    
    def seq2mat(self, seq, max_len=110):
        mat = torch.tensor(list(map(lambda x: self.base2vec[x], seq)), dtype=torch.float32)
        return mat

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        _, seq, target = self.records[idx]
        X = self.seq2mat(seq)
        y = torch.tensor(float(target), dtype=torch.float32)
        y_long = torch.tensor(target, dtype=torch.long)
        
        return X, y, y_long


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dir: str = "data/train_v2.csv",
        test_dir: str = "data/test_v2.csv",
        batch_size: int = 256, 
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.dataset = OneHotDataset
    
    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            train_df = pd.read_csv(self.hparams.train_dir)
            train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
            self.train_data = self.dataset(train_df)
            self.val_data = self.dataset(val_df)
        
        if stage == "test" or stage == None:
            test_df = pd.read_csv(self.hparams.test_dir)
            self.test_data = self.dataset(test_df)
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.test_data, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )