import torch
import torch.nn as nn


def _block(in_f: int, out_f: int, dropout: float = 0.1) -> list:
    return [
        nn.Linear(in_f, out_f),
        nn.BatchNorm1d(out_f),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    ]



class BaseRegressionNet(nn.Module):
    def __init__(self, in_features = 8):
        super().__init__()
        self.network = nn.Sequential(
            *_block(in_features, 64),
            *_block(64, 32),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)



class MTLAutoencoderRegressor(nn.Module):
    def __init__(self, in_features = 8, latent_dim = 4):
        super().__init__()

        self.encoder = nn.Sequential(
            *_block(in_features, 64),
            *_block(64, 32),
            nn.Linear(32, 4)
        )

        self.decoder = nn.Sequential(
            *_block(latent_dim, 32),
            *_block(32, 64),
            nn.Linear(64, 8)
        )

        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        z = self.encoder(x)

        x_hat = self.decoder(z)
        y_hat = self.regressor(z)

        return x_hat, y_hat



class BinaryClassifierMLP(nn.Module):
    def __init__(self, in_features=20):
        super().__init__()
        self.network = nn.Sequential(
            *_block(in_features, 128),
            *_block(128, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)