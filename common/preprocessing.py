"""common.preprocessing — train-fold-only StandardScaler + LabelEncoder.

Owned by the runner (common.base) so every one of the 30 models gets identical
preprocessing with no train/test leakage and no per-module duplication. Models
that need their OWN internal scaling for a geometry step (e.g. MDO's PCA) may
fit an extra scaler inside fit() on the already-scaled X they receive.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Preprocessor:
    """Fit on the TRAIN fold only; transform both train and test identically."""

    def __init__(self, scale=True):
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.encoder = LabelEncoder()
        self.classes_ = None      # original label values (for confusion-matrix ticks)

    def fit_transform(self, X_train, y_train):
        y_enc = self.encoder.fit_transform(np.asarray(y_train).ravel())
        self.classes_ = self.encoder.classes_
        if self.scale:
            X_sc = self.scaler.fit_transform(np.asarray(X_train, dtype=float))
        else:
            X_sc = np.asarray(X_train, dtype=float)
        return X_sc, y_enc

    def transform_X(self, X):
        X = np.asarray(X, dtype=float)
        return self.scaler.transform(X) if self.scale else X

    def transform_y(self, y):
        return self.encoder.transform(np.asarray(y).ravel())
