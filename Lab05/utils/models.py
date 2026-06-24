import torch
import torch.nn as nn

class wideMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 128)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(128, 128) for _ in range(3)
        ])
        self.output = nn.Linear(128, 7)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output(x)

class deepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 64)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(64, 64) for _ in range(12)
        ])
        self.output = nn.Linear(64, 7)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output(x)

class pyramidMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(16, 32)
        layer_dims = [32, 64, 128, 128, 128, 64, 32, 16]

        self.hidden_layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)
        ])

        self.output = nn.Linear(16, 7)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output(x)