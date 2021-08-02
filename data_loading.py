import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms


cifar10_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])


class NormClfDataset(Dataset):
    def __init__(self, X, y, transforms=cifar10_transforms):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self, ):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        y = self.y[idx].astype(int)
        if self.transforms is not None:
            X = self.transforms(X)

        return X, torch.tensor(y)


def safe_load_numpy(name):
    name_npy = name + ".npy"
    name_csv = name + ".csv"
    if os.path.exists(name_npy):
        return np.load(name_npy)
    else:
        if not os.path.exists(name_csv):
            os.system(f"gsutil -m cp -r gs://neurips2021_bdl_competition/{name}.csv .")
        d = np.loadtxt(name_csv)
        np.save(name_npy, d)
        return d
