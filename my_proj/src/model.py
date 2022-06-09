import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC


class CNN(nn.Module):
    def __init__(
        self,
        kernel_size=5,
        out_channels=64,
        pool_size=2,
        hidden_dim=64
    ):
        super().__init__()
        conv_out_len = 21 - kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = out_channels * pool_out_len
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=out_channels, kernel_size=kernel_size, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x = self.flat(x)
        return self.mlp(x).squeeze(-1)
    
    
class RNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(21 * hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (N, L, C)
        x, _ = self.lstm(x)
        x = self.flat(x)
        return self.mlp(x).squeeze(-1)
    
    
class CNNRNN(nn.Module):
    def __init__(
        self,
        kernel_size=5,
        out_channels=64,
        pool_size=2,
        lstm_hidden_dim=64,
        mlp_hidden_dim=64
    ):
        super().__init__()
        conv_out_len = 21 - kernel_size + 1
        pool_out_len = int(1 + ((conv_out_len - pool_size) / pool_size))
        fc_input_dim = lstm_hidden_dim * 2 * pool_out_len
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=out_channels, kernel_size=kernel_size, padding="valid"),
            nn.ReLU(),
            nn.MaxPool1d(pool_size)
        )
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.flat = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(fc_input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (N, L, C)
        x = x.transpose(1, 2)  # (N, C, L)
        x = self.conv_block(x)  # (N, C, L)
        x = x.transpose(1, 2)  # (N, L, C)
        x, _ = self.lstm(x)
        x = self.flat(x)
        return self.mlp(x).squeeze(-1)
    
    
class MyNet(pl.LightningModule):
    def __init__(
        self,
        net,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        
        self.criterion = nn.BCELoss()
        
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        
        self.train_auroc = AUROC()
        self.val_auroc = AUROC()
        self.test_auroc = AUROC()
        
    def forward(self, x):
        return self.net(x)
    
    def step(self, batch):
        X, y, y_long = batch
        pred = self(X)
        loss = self.criterion(pred, y)

        return loss, pred, y_long
    
    def training_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        self.train_acc(pred, y)
        self.train_auroc(pred, y)
        metrics = {"train/loss": loss, "train/accuracy": self.train_acc, "train/auroc": self.train_auroc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        self.val_acc(pred, y)
        self.val_auroc(pred, y)
        metrics = {"val/loss": loss, "val/accuracy": self.val_acc, "val/auroc": self.val_auroc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        self.test_acc(pred, y)
        self.test_auroc(pred, y)
        metrics = {"test/loss": loss, "test/accuracy": self.test_acc, "test/auroc": self.test_auroc}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        _, preds, _ = self.step(batch)
        
        return preds
    
    def on_predict_epoch_end(self, outputs):
        whole_preds = torch.cat(outputs[0])
        return whole_preds
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                                lr=self.hparams.learning_rate, 
                                weight_decay=self.hparams.weight_decay)