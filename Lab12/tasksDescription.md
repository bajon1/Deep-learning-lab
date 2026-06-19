# Lab 12: Varia

---

## Task 1: Multiple-Task Learning

**Goal:** Solve a regression problem in a multiple-task learning (MTL) scheme — autoencoder combined with a regressor.

1. Generate a correlated regression dataset with `generate_correlated_regression_data(n_samples=3000, random_state=42)`. The 8 input features are linear combinations of 8 independent base variables with small noise. The target `y` depends on non-linear transformations of those features.
2. Split the data into train / val / test sets. Create `Dataset` and `DataLoader` instances.
3. Implement and train the baseline regression model `BaseRegressionNet`:

```python
BaseRegressionNet(
  (network): Sequential(
    (0): Linear(in_features=8, out_features=64)
    (1): BatchNorm1d(64)
    (2): ReLU()
    (3): Dropout(p=0.1)
    (4): Linear(in_features=64, out_features=32)
    (5): BatchNorm1d(32)
    (6): ReLU()
    (7): Dropout(p=0.1)
    (8): Linear(in_features=32, out_features=1)
  )
)
```

4. Implement the MTL model `MTLAutoencoderRegressor`. The encoder maps input `X` to a latent vector `z` of dimension 4. The decoder reconstructs `X` from `z`. The regressor predicts `y` from `z`. The model returns `(X_hat, y_hat)`:

```python
MTLAutoencoderRegressor(
  (encoder): Sequential(
    (0): Linear(in_features=8, out_features=64)
    (1): BatchNorm1d(64)  (2): ReLU()  (3): Dropout(p=0.1)
    (4): Linear(in_features=64, out_features=32)
    (5): BatchNorm1d(32)  (6): ReLU()  (7): Dropout(p=0.1)
    (8): Linear(in_features=32, out_features=4)
  )
  (decoder): Sequential(
    (0): Linear(in_features=4, out_features=32)
    (1): BatchNorm1d(32)  (2): ReLU()  (3): Dropout(p=0.1)
    (4): Linear(in_features=32, out_features=64)
    (5): BatchNorm1d(64)  (6): ReLU()  (7): Dropout(p=0.1)
    (8): Linear(in_features=64, out_features=8)
  )
  (regressor): Sequential(
    (0): Linear(in_features=4, out_features=16)
    (1): ReLU()  (2): Dropout(p=0.1)
    (3): Linear(in_features=16, out_features=1)
  )
)
```

5. Train and evaluate both models. The MTL loss is a weighted sum of reconstruction loss and regression loss:

```python
reconstruction_criterion = nn.MSELoss()
regression_criterion     = nn.MSELoss()

X_hat, y_hat = model(X_batch)
recon_loss   = reconstruction_criterion(X_hat, X_batch)
reg_loss     = regression_criterion(y_hat, y_batch)
total_loss   = ALPHA_RECON * recon_loss + BETA_REG * reg_loss
```

Use the following training settings:

```python
BATCH_SIZE   = 128
EPOCHS       = 1000
LR           = 1e-3
WEIGHT_DECAY = 1e-4
ALPHA_RECON  = 0.4
BETA_REG     = 1.0
```

Compare both models across datasets generated with different random seeds using **MSE**, **MAE**, and **R²**.

**Assignment:** Implement, train and compare single-task and multiple-task learning regression models.

---

## Task 2: Focal Loss

**Goal:** Use Focal Loss as an alternative to cross-entropy for training a classifier on a strongly imbalanced dataset.

1. Generate a strongly imbalanced binary classification dataset (95% class 0, 5% class 1):

```python
from sklearn.datasets import make_classification

IMBALANCE_RATIO = 0.95

def make_imbalanced_dataset(n_samples=12000, n_features=20, imbalance_ratio=0.95, seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=6,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[imbalance_ratio, 1 - imbalance_ratio],
        class_sep=0.9,
        flip_y=0.02,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.float32)
```

2. Implement `BinaryFocalLoss`:

```python
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

class BinaryFocalLoss(nn.Module):
    """
    FL = -alpha_t * (1 - p_t)^gamma * BCE
    where:
      p_t     = p if y=1, else 1-p
      alpha_t = alpha if y=1, else 1-alpha
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs   = torch.sigmoid(logits)
        pt      = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1 - self.alpha))
        loss    = alpha_t * (1 - pt).pow(self.gamma) * bce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
```

3. Implement datasets, dataloaders, an MLP classifier, training loop, and a prediction function (use the best checkpoint from validation).
4. Implement evaluation metrics:

```python
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "pr_auc":    average_precision_score(y_true, y_prob),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
```

5. Train with `nn.BCEWithLogitsLoss` and with `BinaryFocalLoss`. Compare results on the test set.

**Assignment:** Implement and train a classifier on a strongly imbalanced dataset — compare classification quality for BCE and Focal Loss.

---

## Task 3: Soft Label Loss

**Goal:** Use Label Smoothing BCE as an alternative to cross-entropy on the same imbalanced dataset.

1. Implement `LabelSmoothingBCE`:

```python
class LabelSmoothingBCE(nn.Module):
    """
    Replaces hard binary targets {0, 1} with smoothed targets:
      y_smoothed = y * (1 - eps) + (1 - y) * eps
    Prevents overconfidence; acts as a regulariser for noisy labels.
    Typical eps range: 0.05 – 0.15
    """
    def __init__(self, eps=0.1, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits, targets):
        targets          = targets.float()
        smoothed_targets = targets * (1.0 - self.eps) + (1.0 - targets) * self.eps
        return F.binary_cross_entropy_with_logits(logits, smoothed_targets,
                                                   reduction=self.reduction)
```

2. Train and evaluate analogously to Task 2, for several values of the smoothing coefficient, e.g. `eps ∈ {0.05, 0.10, 0.15}`.

**Assignment:** Implement and train a classifier on a strongly imbalanced dataset — compare classification quality for BCE and Soft Label Loss.

---

## Task 4: Tversky Loss

**Goal:** Use Tversky Loss as an alternative to cross-entropy on the same imbalanced dataset. Tversky Loss allows asymmetric penalisation of false positives and false negatives.

1. Implement `TverskyLoss`:

```python
class TverskyLoss(nn.Module):
    """
    TI   = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    Loss = 1 - TI

    Special cases:
      alpha=0.5, beta=0.5  →  Dice Loss
      alpha=0.0, beta=1.0  →  pure Recall loss

    For imbalanced data, beta > alpha penalises FN more heavily → higher recall.
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, from_logits=True):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits, targets):
        if self.from_logits:
            probs = torch.sigmoid(logits.squeeze(1) if logits.dim() == 2 else logits)
        else:
            probs = logits.squeeze(1)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index
```

2. Train and evaluate analogously to Task 2, for several `(alpha, beta)` configurations:

| alpha | beta | effect                         |
|-------|------|--------------------------------|
| 0.5   | 0.5  | Dice Loss (symmetric)          |
| 0.3   | 0.7  | recall-biased                  |
| 0.1   | 0.9  | strong recall (penalise FN)    |

**Assignment:** Implement and train a classifier on a strongly imbalanced dataset — compare classification quality for BCE and Tversky Loss.

---

## Task 5: Inverse Problem

**Goal:** Use a trained regression model and an optimiser to solve the inverse problem — given a target output value `y*`, find the input `X` such that `model(X) ≈ y*`.

1. Load the `diabetes` dataset and split into train / val / test:

```python
from sklearn.datasets import load_diabetes

diabetes      = load_diabetes()
X, y          = diabetes.data.astype(np.float32), diabetes.target.astype(np.float32)
feature_names = diabetes.feature_names
```

2. Implement, train and evaluate a regression model on the diabetes dataset.
3. Implement inverse prediction using `scipy.optimize.minimize`. Try `n_trials` random starting points within training-data bounds and return the best result:

```python
from scipy.optimize import minimize

def inverse_prediction_minimize(model, y_target, X_train, method="L-BFGS-B", n_trials=5):
    """
    Find X such that model(X) ≈ y_target.
    Parameters
    ----------
    model    : trained regression model
    y_target : float, target output value
    X_train  : ndarray, training data used to determine feature bounds
    method   : optimisation algorithm
    n_trials : number of random restarts
    Returns
    -------
    best_x    : ndarray, best found input vector
    best_rmse : float, RMSE at best_x
    """
    def objective(x):
        # compute y_pred using the regressor model and return (y_pred - y_target)**2
        ...

    x_min  = X_train.min(axis=0)
    x_max  = X_train.max(axis=0)
    bounds = list(zip(x_min, x_max))

    best_x, best_error = None, float("inf")
    for trial in range(n_trials):
        x0     = ...  # random starting point within bounds
        result = minimize(objective, x0, method=method, bounds=bounds,
                          options={"maxiter": 1000})
        if result.success and result.fun < best_error:
            best_x, best_error = result.x, result.fun

    return best_x, np.sqrt(best_error)
```

4. Test on the test set:

```python
y_target         = np.median(y_test)
x_inverse, error = inverse_prediction_minimize(model, y_target, X_train, n_trials=10)
y_check          = model(torch.from_numpy(x_inverse.astype(np.float32)).unsqueeze(0).to(DEVICE)).item()
```

5. Propose and implement an Optuna-based approach to solve the inverse problem. Each trial suggests a candidate `X` within training-data bounds; the objective is the squared prediction error.

**Assignment:** Implement inverse prediction based on a trained regression model.