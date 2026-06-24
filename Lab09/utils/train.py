import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from .model import *


def fold_train(X_trainval, y_trainval, model, n_splits=5, lr=1e-3, EPOCHS=10, device='cpu'):
    fold_models = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval, y_trainval)):

        best_val_loss = np.inf
        train_losses = []
        val_losses = []

        X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
        X_val, y_val = X_trainval[val_idx].to(device), y_trainval[val_idx].to(device)

        fold_model = copy.deepcopy(model).to(device)
        optim = torch.optim.Adam(fold_model.parameters(), lr=lr)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=1024,
            shuffle=True,
            pin_memory=(device == 'cpu')
        )

        for epoch in range(EPOCHS):

            fold_model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optim.zero_grad()
                loss = criterion(fold_model(X_batch), y_batch)
                loss.backward()
                optim.step()
                train_loss += loss.item() * len(X_batch)

            train_losses.append(train_loss / len(X_train))

            fold_model.eval()
            with torch.no_grad():
                val_loss = criterion(fold_model(X_val), y_val).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model_state': fold_model.state_dict(),
                    'optim_state': optim.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'train_losses': train_losses.copy(),
                    'val_losses': val_losses.copy()
                }

        if device == 'mps':
            torch.mps.empty_cache()
        elif device == 'cuda':
            torch.cuda.empty_cache()

        del fold_model, optim, X_train, y_train, X_val, y_val

        fold_models.append(checkpoint)
        print(f"Fold {fold+1}/{n_splits} — best val loss: {best_val_loss:.4f}")

    return fold_models