import torch
import torch.nn as nn
import optuna
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class optunaMLP(nn.Module):
    def __init__(self, n_hidden, hidden_dim, input_dim, output_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.output(x)


def objective(trial, X_trainval, y_trainval, input_dim, output_dim, study_writer):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 4096)
    n_hidden = trial.suggest_int('n_hidden', 1, 16)
    hidden_dim = trial.suggest_int('hidden_dim', 4, 512)

    val_losses = []
    kfold = KFold(n_splits=5, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_trainval)):
        X_train, X_val = X_trainval[train_idx], torch.tensor(X_trainval[val_idx]).to(device)
        y_train, y_val = y_trainval[train_idx], torch.tensor(y_trainval[val_idx]).long().to(device)

        model = optunaMLP(n_hidden, hidden_dim, input_dim, output_dim).to(device)

        val_loss = _optuna_train(model, device, X_train, X_val, y_train, y_val, lr, batch_size)
        val_losses.append(val_loss)

    mean_val_loss = np.mean(val_losses)

    study_writer.add_scalar('Study/val_loss', mean_val_loss, trial.number)
    study_writer.add_scalars('Study/hparams', {'lr': lr, 'n_hidden': n_hidden, 'hidden_dim': hidden_dim,}, trial.number)
    study_writer.flush()

    return mean_val_loss

def _optuna_train(model, device, X_train, X_val, y_train, y_val, lr, batch_size):
    best_val_loss = float('inf')

    train_loader = torch.utils.data.DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train).long()),
                                               batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(100):
        val_loss = 0
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = criterion(model(X_batch), y_batch)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss