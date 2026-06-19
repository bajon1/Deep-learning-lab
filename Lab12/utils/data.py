import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def make_loaders(X, y, batch_size, test_size=0.2, squeeze_y=False):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val         = train_test_split(X_trainval, y_trainval, test_size=test_size)

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True,  squeeze_y=squeeze_y)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False, squeeze_y=squeeze_y)
    test_loader  = _make_loader(X_test,  y_test,  batch_size, shuffle=False, squeeze_y=squeeze_y)

    return train_loader, val_loader, test_loader


def _make_loader(X, y, batch_size, shuffle=False, squeeze_y=False):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    if not squeeze_y:
        y_t = y_t.unsqueeze(1)

    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)