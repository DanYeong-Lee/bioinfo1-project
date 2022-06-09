import os
import random
import numpy as np
import torch

from src.datamodule import MyDataModule
from src.model import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    

def train_test(model_class, seed):
    name = f"{model_class.__name__}_{seed}"
    
    seed_everything(seed)
    
    dm = MyDataModule()
    model = MyNet(model_class())
    early_stop = EarlyStopping(monitor="val/loss", mode="min", patience=3)
    model_ckpt = ModelCheckpoint(dirpath="ckpts", filename=name, monitor="val/loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=30, gpus=[0], callbacks=[early_stop, model_ckpt])
    trainer.fit(model, dm)
    results = trainer.test(model, dm, ckpt_path=f"ckpts/{name}.ckpt")
    
    test_acc = results[0]["test/accuracy"]
    test_auroc = results[0]["test/auroc"]
    
    return test_acc, test_auroc


def train_Nseed(model_class, N=20):
    seeds = random.sample(range(1000), N)
    accs = []
    aurocs = []
    for seed in seeds:
        acc, auroc = train_test(model_class, seed)
        accs.append(acc)
        aurocs.append(auroc)
    
    name = model_class.__name__
    accs = np.array(accs)
    aurocs = np.array(aurocs)
    
    with open("results.csv", "a") as f:
        f.write(f"{name},{accs.mean()},{accs.std()},{aurocs.mean()},{aurocs.std()}\n")
        
        
if __name__ == "__main__":
    train_Nseed(CNN, 20)
    train_Nseed(RNN, 20)
    train_Nseed(CNNRNN, 20)