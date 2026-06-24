# Lab 9: Probability Calibration

---

## Task 1: Probability calibration on a toy dataset

#### Goal: Implement temperature scaling calibration for a neural network classifier on a synthetic dataset.

1. Generate a classification toy dataset using `make_classification` (5000 samples, 20 features). Convert to `float32` tensors and split into train/val, calibration, and test subsets.
2. Train a neural network classifier using cross-validation (5 folds). Save the best model checkpoint per fold — this produces an ensemble of 5 models.
3. Implement a `TemperatureScaler` module that wraps a trained model and divides its logits by a learnable scalar parameter `T`. Implement `fit_temperature` that optimizes `T` using `LBFGS` and `CrossEntropyLoss` on the calibration set. Fit a separate scaler for each of the 5 ensemble models.
4. Compute and plot calibration curves (fraction of positives vs. mean predicted probability, 10 bins) for each model before and after temperature scaling using `calibration_curve` from scikit-learn.
5. Compute Brier score before and after calibration using `brier_score_loss` for each ensemble model.

**Assignment:** Calibrate a classifier on the toy dataset using temperature scaling. Present results as calibration curves and Brier score values.

---

## Task 2: Uncertainty estimation before and after calibration

#### Goal: Assess the effect of probability calibration on prediction uncertainty.

1. Evaluate ensemble accuracy before and after calibration on the test set. Compare the two using McNemar's test.
2. Compute and plot the distribution of **total uncertainty** (predictive entropy) before and after calibration:
$$H = -\sum_c \bar{p}_c \log \bar{p}_c$$
where $\bar{p}_c$ is the mean predicted probability for class $c$ across ensemble members.
3. Compute and plot the distribution of **epistemic uncertainty** (mutual information between predictions and model parameters) before and after calibration:
$$U_{\text{epistemic}} = H(\bar{p}) - \frac{1}{K}\sum_{k=1}^K H(p^{(k)})$$
4. Compute and plot the distribution of **aleatoric uncertainty** (expected entropy of individual model predictions) before and after calibration:
$$U_{\text{aleatoric}} = \frac{1}{K}\sum_{k=1}^K H(p^{(k)})$$

**Assignment:** Compare classifier accuracy and uncertainty distributions before and after probability calibration.

---

## Task 3: Scikit-learn wrapper for a PyTorch model

#### Goal: Implement a scikit-learn-compatible wrapper for a PyTorch model and use it with scikit-learn calibration methods.

1. Implement `MyWrapper` as a subclass of `ClassifierMixin` and `BaseEstimator`. The constructor must accept at minimum: `model`, `classes`, `lr`, `epochs`, `batch_size`, `device`, and `is_fitted`. The wrapper must support both pre-trained and untrained PyTorch models (controlled by the `is_fitted` flag).
2. Implement the `fit`, `predict`, `predict_proba`, and `score` methods so that the wrapper integrates with standard scikit-learn utilities such as `cross_val_score` and `confusion_matrix`.
3. Wrap a trained model with `is_fitted=True` and verify it works with `cross_val_score`, `confusion_matrix`, and `accuracy_score`.
4. Use `CalibratedClassifierCV` with `FrozenEstimator` to calibrate the wrapped model using all three available methods: `sigmoid`, `isotonic`, and `temperature` (available from scikit-learn 1.8.0). For each method, compute calibration curves and Brier score on the validation set.

**Assignment:** Calibrate a trained PyTorch model using scikit-learn infrastructure. Compare calibration quality across all three methods using Brier score.

---

## Task 4: Calibration on a real-world dataset

#### Goal: Train and calibrate a classifier on a real-world dataset using the methods from Tasks 1–3.

1. Select a real-world classification dataset (e.g. from OpenML, `ucimlrepo`, or scikit-learn). Preprocess features and targets, convert to tensors, and split into train/val/calibration/test subsets.
2. Train an ensemble of models using cross-validation as in Task 1. Apply temperature scaling calibration using `fit_temperature` from Task 1.
3. Wrap the trained model using `MyWrapper` from Task 3 and apply `CalibratedClassifierCV` calibration methods.
4. Compare accuracy, calibration curves, Brier score, and uncertainty distributions (total, epistemic, aleatoric) before and after calibration.

**Assignment:** Train and calibrate a classifier on a real-world dataset. Evaluate prediction accuracy and uncertainty distributions before and after calibration.