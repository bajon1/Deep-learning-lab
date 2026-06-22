import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import r2_score

class LitModel(pl.LightningModule):
    def __init__(self, hidden_layers, input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.input_layer = nn.Linear(in_features=input_dim, out_features=hidden_layers[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_features=hidden_layers[i], out_features=hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)
        ])
        self.output_layer = nn.Linear(in_features=hidden_layers[-1], out_features=output_dim)

        self.activation = nn.ReLU()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        return self.output_layer(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()

        train_loss = self.criterion(y_hat, y)
        self.log('train_loss', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()

        val_loss = self.criterion(y_hat, y)
        val_r2 = r2_score(y_hat, y)

        self.log('val_loss', val_loss)
        self.log('val_r2', val_r2)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze()

        test_loss = self.criterion(y_hat, y)
        test_r2 = r2_score(y_hat, y)

        self.log('test_loss', test_loss)
        self.log('test_r2', test_r2)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)