from typing import List
import itertools
import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_kmer_dataset(k=4):
    kmers = ["".join(v) for v in itertools.product(*["ACGT"] * k)]
    kmer2idx = {kmer:i for i, kmer in enumerate(kmers)}
    
    def to_kmer_vector(seq, k):
        v = np.zeros(4 ** k)
        for i in range(len(seq) - k + 1):
            kmer = seq[i: i+k]

            v[kmer2idx[kmer]] += 1

        return v

    def df_to_kmer(df, k):
        kmer_vectors = []
        for seq in df.seq.values:
            kmer_vectors.append(to_kmer_vector(seq, k=k))

        kmer_vectors = np.vstack(kmer_vectors)

        return kmer_vectors
    
    train_df = pd.read_csv("data/train.csv")
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    test_df = pd.read_csv("data/test.csv")
    
    train_X = df_to_kmer(train_df, k=k)
    val_X = df_to_kmer(val_df, k=k)
    test_X = df_to_kmer(test_df, k=k)

    train_y = train_df.target.to_numpy()
    val_y = val_df.target.to_numpy()
    test_y = test_df.target.to_numpy()
    
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def train_test(model_class, train_data, test_data, seed):
    seed_everything(seed)
    
    clf = make_pipeline(StandardScaler(), model_class())
    clf.fit(*train_data)
    
    acc = clf.score(*test_data)
    auroc = roc_auc_score(test_data[1], clf.predict_proba(test_data[0])[:, 1])
    
    return acc, auroc
    
    
def train_Nseeds(model_class, train_data, test_data, N=20):
    seeds = random.sample(range(1000), N)
    accs = []
    aurocs = []
    for seed in seeds:
        acc, auroc = train_test(model_class, train_data, test_data, seed)
        accs.append(acc)
        aurocs.append(auroc)
    
    name = model_class.__name__
    accs = np.array(accs)
    aurocs = np.array(aurocs)
    
    with open("results.csv", "a") as f:
        f.write(f"{name},{accs.mean()},{accs.std()},{aurocs.mean()},{aurocs.std()}\n")
        
        
if __name__ == "__main__":
    with open("results.csv", "w") as f:
        f.write(f"Model,acc_mean,acc_std,auroc_mean,auroc_std\n")
    train_data, val_data, test_data = get_kmer_dataset(4)
    train_Nseeds(LogisticRegression, train_data, test_data, N=20)
    train_Nseeds(SVC, train_data, test_data, N=20)
    train_Nseeds(RandomForestClassifier, train_data, test_data, N=20)