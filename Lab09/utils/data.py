import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class MyWrapper(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, model, classes, batch_size=256, lr=1e-3,
                 epochs=10, device='cpu', is_fitted=False):
        self.model = model
        self.classes = classes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.is_fitted = is_fitted

    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        self.model_  = copy.deepcopy(self.model).to(self.device)

        if self.is_fitted:
            self.classes_ = np.asarray(self.classes)
            self.is_fitted_ = True
            return self

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model_(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X.to(self.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
