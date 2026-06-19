import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def make_loaders(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y,)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval,)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader, X_train, y_train


def _make_loader(X, y, batch_size, shuffle = False):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)