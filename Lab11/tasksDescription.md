# Lab 11: Pytorch Lightning

---

## Task 1: Regression network training and testing with PyTorch Lightning

#### Goal: Implement a regression MLP using PyTorch Lightning on a toy dataset.

1. Install the required libraries: `pip install pytorch-lightning` and `pip install ray[tune]`.
2. Implement a `Dataset` subclass (`MyDataset`) that stores input features and targets and returns individual samples via `__getitem__`.
3. Implement a `LightningDataModule` (`MyDataModule`) that generates a regression toy dataset using `make_regression`, converts it to `float32` tensors, splits it into train/val/test subsets, and exposes them via `train_dataloader`, `val_dataloader`, and `test_dataloader`.
4. Implement a `LightningModule` (`LitModel`) with a 3-layer MLP. Use `self.save_hyperparameters()` in `__init__`. Implement `training_step`, `validation_step`, and `test_step` — each should compute MSE loss; validation and test steps should also compute and log R² score. Implement `configure_optimizers` returning an Adam optimizer using `self.hparams.lr`.
5. Instantiate a `Trainer` with `max_epochs` and appropriate `accelerator` settings and run `trainer.fit()` followed by `trainer.test()`.
6. Add an `EarlyStopping` callback (monitoring `val_loss` with patience 5) and a `ModelCheckpoint` callback (saving the top 3 checkpoints by `val_loss` plus the last one). Re-run training with both callbacks active.
7. Define a `train_model(config)` function for Ray Tune that instantiates `LitModel` with sampled `hidden_size` and `lr` values and trains it using a `TuneReportCallback`. Run the hyperparameter search using `tune.run()` with `tune.choice` for `hidden_size` and `tune.loguniform` for `lr` over 10 samples, then print the best config.

**Assignment:** Implement and run hyperparameter optimization for a regression network on the toy dataset using PyTorch Lightning.

---

## Task 2: PyTorch Lightning for regression — two real-world datasets

#### Goal: Implement and optimize a regression MLP on real-world datasets using PyTorch Lightning.

1. Reuse the `MyDataset`, `MyDataModule`, and `LitModel` classes from Task 1.
2. Replace the `make_regression` toy data in `MyDataModule` with two real-world regression datasets (e.g. from `ucimlrepo`, OpenML, or sklearn). Adjust `input_size` and `output_dim` accordingly.
3. For each dataset, run the full training pipeline including callbacks and hyperparameter optimization as implemented in Task 1.

**Assignment:** Implement and run hyperparameter optimization for a regression network on two real-world datasets using PyTorch Lightning.

---

## Task 3: PyTorch Lightning for classification — two real-world datasets

#### Goal: Implement and optimize a classification MLP on real-world datasets using PyTorch Lightning.

1. Reuse the structure from Task 1, adapting `LitModel` for classification: replace MSELoss with `CrossEntropyLoss`, update `training_step`, `validation_step`, and `test_step` to log accuracy (and optionally F1 score) instead of R².
2. Update `MyDataModule` to load two real-world classification datasets. Ensure targets are of type `torch.long`.
3. For each dataset, run the full training pipeline including `EarlyStopping`, `ModelCheckpoint`, and Ray Tune hyperparameter search as in Task 1.

**Assignment:** Implement and run hyperparameter optimization for a classification network on two real-world datasets using PyTorch Lightning.