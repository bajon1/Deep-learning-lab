## Task 1: TensorBoard installation and basic SummaryWriter operations

#### Goal: Get familiar with TensorBoard and SummaryWriter usage.

1. Install TensorBoard (`conda install -c conda-forge tensorboard` or `pip install tensorboard`). If needed, also install a compatible protobuf version: `pip install "protobuf==4.25.3"`.
2. Install GraphViz and torchviz for model architecture visualization.
3. Launch TensorBoard from the terminal with `tensorboard --logdir runs` and open it in the browser at `localhost:6006/`.
4. Create two `SummaryWriter` instances logging to different subdirectories (`runs/test1` and `runs/test2`).
5. For each writer: define a small `nn.Sequential` model, add its computation graph using `writer.add_graph()`, and log a scalar (or multiple scalars) over a number of steps using `writer.add_scalars()`.
6. Call `writer.flush()` after each logging step to ensure data is written to disk.

**Assignment:** Test the functionality of TensorBoard and SummaryWriter. Handle logging to different subdirectories and verify that both runs appear correctly in the TensorBoard interface.

---

## Task 2: Logging and visualizing training progress

#### Goal: Use TensorBoard and SummaryWriter to log and visualize the loss function on the training and validation sets.

1. Reuse a model and dataset from a previous lab (e.g. the regression or classification MLP from Lab 4).
2. Create a `SummaryWriter` and log both training loss and validation loss at each epoch using `writer.add_scalars()`, storing them under the same `main_tag` (e.g. `'loss'`) with keys `'training'` and `'validation'`.
3. Call `writer.flush()` at the end of each epoch.
4. Open TensorBoard and inspect the loss curves for both splits.

**Assignment:** Implement loss function logging for both training and validation sets using SummaryWriter. Visualize the training progress in TensorBoard and verify that the curves behave as expected (e.g. decreasing loss, no divergence).

---

## Task 3: Logging and visualizing hyperparameter optimization

#### Goal: Use TensorBoard and SummaryWriter to log and visualize hyperparameter optimization using the Optuna library.

1. Define an Optuna objective function that samples hyperparameters (e.g. learning rate, batch size, number of layers) and trains a model for a fixed number of epochs.
2. For each Optuna trial, create a dedicated `SummaryWriter` (e.g. `runs/trial_{trial.number}`) and log the per-epoch loss using `writer.add_scalars()`.
3. After training, log the trial's hyperparameters and final metrics using `writer.add_hparams(hparams_dict, metrics_dict)` and call `writer.flush()`.
4. Run the Optuna study and inspect the results in TensorBoard — check both the scalar loss curves per trial and the HParams tab.
5. Export the optimization results (trial number, hyperparameters, final metric) to a CSV file using `study.trials_dataframe().to_csv()`.

**Assignment:** Implement loss and hyperparameter logging for each Optuna trial using SummaryWriter. Visualize the optimization progress in TensorBoard and export the results to CSV.

---

## Task 4: Logging images, plots, and histograms

#### Goal: Use SummaryWriter to log images, matplotlib figures, and weight histograms.

1. Log an image (or a batch of images) using `writer.add_image()`. The image tensor must have shape `[C, H, W]` and values in `[0, 1]`. You can use a sample from a torchvision dataset or a randomly generated tensor.
2. Create a matplotlib figure (e.g. a plot of the loss curve or a scatter plot of predictions vs. targets) and log it using `writer.add_figure()`.
3. After each training epoch, log the weight distributions of the model's layers using `writer.add_histogram()` — iterate over `model.named_parameters()` and add a histogram for each parameter tensor.
4. Call `writer.flush()` after all logging calls and inspect the Images, Figures, and Histograms tabs in TensorBoard.

**Assignment:** Implement logging of images, matplotlib figures, and weight histograms using SummaryWriter. Verify that all three appear correctly in the corresponding TensorBoard tabs.

---