from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super().__init__()

        self.input = nn.Linear(input_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)
        ])
        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output(x)