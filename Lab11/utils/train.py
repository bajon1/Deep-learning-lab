import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from .models import LitModel
from .dataset import MyDataModule

def train_model(config):
    datamodule = MyDataModule(
        batch_size=config['batch_size'],
        n_samples=config['n_samples'],
        n_features=config['input_dim']
    )

    model = LitModel(hidden_layers=config['hidden_layers'],
                     input_dim = config['input_dim'],
                     output_dim = config['output_dim']
                     )

    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         callbacks=[TuneReportCallback({'loss': 'val_loss'}, on='validation_end')],
                         )

    trainer.fit(model, datamodule=datamodule)