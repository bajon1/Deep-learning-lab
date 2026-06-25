import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .train import get_all_probs


class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.log_temperature = nn.Parameter(torch.ones(1))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
        return logits / self.temperature


def fit_temperature(model, X, y):
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)

    scaler = TemperatureScaler(model).to(device)
    optimizer = optim.LBFGS([scaler.log_temperature], lr=0.01, max_iter=500)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        logits = scaler(X)
        loss = criterion(logits, y)
        loss.backward()
        return loss

    optimizer.step(closure)

    print(f"  Fitted T = {scaler.temperature.item():.4f}")
    return scaler


def entropy(probs):
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def uncertainty_decomposition(models, X, device):
    all_probs = get_all_probs(models, X, device)
    mean_probs = all_probs.mean(axis=0)

    total = entropy(mean_probs)
    aleatoric = np.mean([entropy(all_probs[k]) for k in range(len(models))], axis=0)
    epistemic = total - aleatoric

    return total, epistemic, aleatoric