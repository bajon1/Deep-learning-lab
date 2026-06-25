import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
        return logits / self.temperature


def fit_temperature(model, X, y):
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device)

    scaler = TemperatureScaler(model).to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=500)
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
