import torch
import torch.utils.data as data
import pytorch_lightning as pl
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class MyDataset(data.Dataset):
    def __init__(self, data, outputs):
        self.data = data
        self.outputs = outputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        y = self.outputs[idx]
        return X, y



class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_samples, n_features):
        super().__init__()
        self.batch_size = batch_size

        X, y = make_regression(n_samples=n_samples, n_features=n_features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def setup(self, stage=None):
        X_trainval, X_test, y_trainval, y_test = train_test_split(self.X, self.y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2)

        self.train_loader = MyDataset(X_train, y_train)
        self.val_loader = MyDataset(X_val, y_val)
        self.test_loader = MyDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_loader, batch_size=self.batch_size, shuffle=True, num_workers=9, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_loader, batch_size=self.batch_size, num_workers=9, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_loader, batch_size=self.batch_size, num_workers=9, persistent_workers=True)
