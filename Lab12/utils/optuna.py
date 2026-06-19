import torch
import numpy as np
import optuna
from scipy.optimize import minimize


def inverse_prediction_minimize(model, y_target, X_train, device='cpu', n_trials=5, SEED=42):
    rng    = np.random.default_rng(SEED)
    x_min  = X_train.min(axis=0)
    x_max  = X_train.max(axis=0)
    bounds = list(zip(x_min, x_max))

    def objective(x):
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        return (model(x_t).item() - y_target) ** 2

    best_x, best_error = None, float("inf")
    for _ in range(n_trials):
        x0     = rng.uniform(x_min, x_max).astype(np.float32)
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000}) # type: ignore
        if result.success and result.fun < best_error:
            best_x, best_error = result.x, result.fun

    return best_x, np.sqrt(best_error)



def inverse_prediction_optuna(model, y_target, X_train, device='cpu', n_trials=200):
    x_min      = X_train.min(axis=0)
    x_max      = X_train.max(axis=0)
    n_features = X_train.shape[1]

    def objective(trial):
        x = np.array([
            trial.suggest_float(f"x{i}", float(x_min[i]), float(x_max[i]))
            for i in range(n_features)
        ], dtype=np.float32)
        return (model(torch.from_numpy(x).unsqueeze(0).to(device)).item() - y_target) ** 2

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)

    best_x = np.array([study.best_params[f"x{i}"] for i in range(n_features)], dtype=np.float32)
    return best_x, np.sqrt(study.best_value)