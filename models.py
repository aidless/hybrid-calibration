"""
Model wrappers: Tree Ensembles, MLP, and Calibrated classifiers.

All models expose a uniform interface:
  - fit(X, y, X_val, y_val)
  - predict_proba(X) -> np.ndarray (n_samples, n_classes) for binary & multiclass
  - predict(X) -> np.ndarray (n_samples,)
"""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb

from config import ModelConfig


# ============================================================
# Base Wrapper Interface
# ============================================================

class BaseModel:
    """Uniform interface for all model types."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.model = None
        self.is_fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        """Return (n_samples, n_classes). For binary, shape is (n, 2)."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.cfg.name} not fitted")
        return self.model.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)


# ============================================================
# Tree Ensemble Models
# ============================================================

class RandomForestModel(BaseModel):
    """scikit-learn Random Forest."""

    def fit(self, X, y, X_val=None, y_val=None):
        self.model = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        return self


class XGBoostModel(BaseModel):
    """XGBoost classifier with probability output."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._num_classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        self._num_classes = len(np.unique(y))
        objective = "binary:logistic" if self._num_classes <= 2 else "multi:softprob"

        params = {
            "n_estimators": self.cfg.n_estimators,
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "objective": objective,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if self._num_classes > 2:
            params["num_class"] = self._num_classes

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        return self


class LightGBMModel(BaseModel):
    """LightGBM classifier."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self._num_classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        self._num_classes = len(np.unique(y))
        objective = "binary" if self._num_classes <= 2 else "multiclass"

        params = {
            "n_estimators": self.cfg.n_estimators,
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "objective": objective,
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        if self._num_classes > 2:
            params["num_class"] = self._num_classes

        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X, y,
            eval_set=eval_set,
        )
        self.is_fitted = True
        return self


# ============================================================
# MLP (Neural Network proxy for LLM classifier head)
# ============================================================

class MLPEmbedModel(BaseModel):
    """Simple MLP trained on frozen LLM embeddings."""

    def fit(self, X, y, X_val=None, y_val=None):
        # When external val set is provided, disable internal early stopping
        # and just train for fixed epochs
        use_early_stopping = X_val is None

        self.model = MLPClassifier(
            hidden_layer_sizes=(self.cfg.hidden_dim, self.cfg.hidden_dim // 2),
            activation="relu",
            solver="adam",
            alpha=1e-4,              # L2 regularization
            batch_size=min(64, len(X)),
            learning_rate_init=1e-3,
            max_iter=self.cfg.mlp_epochs,
            early_stopping=use_early_stopping,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )

        self.model.fit(X, y)
        self.is_fitted = True
        return self


# ============================================================
# Calibrated Models
# ============================================================

class CalibratedModel(BaseModel):
    """
    Wraps a base model with post-hoc calibration
    (Platt scaling via logistic regression, or Isotonic regression).

    sklearn >=1.6 removed cv="prefit" from CalibratedClassifierCV,
    so we implement calibration manually using the base model's
    predicted probabilities on held-out validation data.
    """

    def __init__(self, cfg: ModelConfig, base_model_instance: BaseModel):
        super().__init__(cfg)
        self.base_model = base_model_instance
        self._method = cfg.calibration_method or "platt"
        self.calibrator = None  # LogisticRegression or IsotonicRegression
        self._n_classes = None

    def fit(self, X, y, X_val=None, y_val=None):
        # Base model must already be fitted by the caller.
        # We fit ONLY the calibrator on held-out validation data.
        if X_val is None or y_val is None:
            raise ValueError(
                "CalibratedModel requires held-out validation data (X_val, y_val). "
                "Calibrating on training data produces overconfident calibrators."
            )
        if not self.base_model.is_fitted:
            raise RuntimeError(
                "Base model must be fitted before CalibratedModel.fit() is called."
            )

        # Get base model probabilities on validation set
        y_prob_val = self.base_model.predict_proba(X_val)
        self._n_classes = y_prob_val.shape[1]

        if self._method == "platt":
            # Platt scaling: logistic regression using predicted probabilities
            # as features. For multiclass, fit one calibrator per class.
            self.calibrator = LogisticRegression(
                penalty=None,  # No regularization (pure Platt)
                solver="lbfgs",
                max_iter=1000,
            )
            self.calibrator.fit(y_prob_val, y_val)

        elif self._method == "isotonic":
            # Isotonic regression: fit per-class calibrator on the
            # predicted probability for each class.
            self.calibrator = []
            for k in range(self._n_classes):
                iso = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds="clip"
                )
                iso.fit(y_prob_val[:, k], (y_val == k).astype(np.float64))
                self.calibrator.append(iso)
        else:
            raise ValueError(f"Unknown calibration method: {self._method}")

        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.cfg.name} not fitted")

        y_prob_base = self.base_model.predict_proba(X)

        if self._method == "platt":
            # LogisticRegression.predict_proba returns calibrated probs
            return self.calibrator.predict_proba(y_prob_base)

        elif self._method == "isotonic":
            # Per-class isotonic transformation
            calibrated = np.zeros_like(y_prob_base)
            for k in range(self._n_classes):
                calibrated[:, k] = self.calibrator[k].transform(y_prob_base[:, k])
            # Re-normalize to sum to 1
            row_sums = calibrated.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return calibrated / row_sums

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ============================================================
# Model Factory
# ============================================================

_MODEL_REGISTRY = {
    "rf": RandomForestModel,
    "xgb": XGBoostModel,
    "lgb": LightGBMModel,
    "mlp": MLPEmbedModel,
}


def create_model(cfg: ModelConfig, base_models: Optional[Dict[str, BaseModel]] = None) -> BaseModel:
    """
    Create a model instance from its config.

    Args:
        cfg: ModelConfig describing the model.
        base_models: Dict of fitted base models (needed for calibrated variants).

    Returns:
        BaseModel instance.
    """
    if cfg.model_type == "calibrated":
        if base_models is None or cfg.base_model not in base_models:
            raise ValueError(
                f"Calibrated model '{cfg.name}' requires base model '{cfg.base_model}' "
                f"to be fitted first. Available: {list(base_models.keys()) if base_models else 'none'}"
            )
        return CalibratedModel(cfg, base_models[cfg.base_model])

    model_cls = _MODEL_REGISTRY[cfg.model_type]
    return model_cls(cfg)
