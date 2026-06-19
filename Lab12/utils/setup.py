import torch
import random
import numpy as np
from sklearn.datasets import make_classification


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def generate_correlated_regression_data(n_samples=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    z = rng.normal(0, 1, size=(n_samples, 8))

    x1 = z[:, 0]
    x2 = z[:, 1]
    x3 = z[:, 2]
    x4 = z[:, 3]
    x5 = z[:, 4]
    x6 = z[:, 5]
    x7 = z[:, 6]
    x8 = z[:, 7]

    noise_small = lambda scale=0.05: rng.normal(0, scale, size=n_samples)
    noise_mid = lambda scale=0.15: rng.normal(0, scale, size=n_samples)

    X = np.column_stack([x1 + x2 + noise_small(),
                         2.0 * x3 - 0.5 * x4 + noise_small(),
                         -1.2 * x5 + noise_small(), x6 + x7 + noise_small(),
                         0.7 * x7 - 0.7 * x8 + noise_small(),
                         1.5 * x1 - 0.8 * x3 + noise_mid(),
                         x2 + x4 + x6 + noise_mid(),
                         0.5 * x5 + 0.5 * x8 + noise_mid(),
    ])

    U = np.column_stack([X[:,0] + np.sin(X[:,1]),
                         X[:,2]*X[:,3] + X[:,4]*X[:,5],
                         -X[:,7]*np.sin(X[:,6])
    ])

    y = (3.5 * U[:, 0] + 0.8 * (U[:, 1] ** 2) + 1.2 * np.sin(U[:, 2]) + rng.normal(0, 0.7, size=n_samples))

    return X.astype(np.float32), y.astype(np.float32)



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
