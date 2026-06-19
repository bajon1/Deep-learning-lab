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