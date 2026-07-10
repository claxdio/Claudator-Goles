from __future__ import annotations

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize


def train_gbt(
    X_train: list[list[float]],
    y_train: list[int],
    X_valid: list[list[float]],
    y_valid: list[int],
) -> lgb.Booster:
    """Trains a LightGBM binary classifier with regularization appropriate
    for a low-thousands-of-matches dataset. Reports validation loss via
    `valid_sets` so training progress is visible, but does not rely on an
    early-stopping callback (its API differs across lightgbm versions) --
    the shallow depth, min_data_in_leaf, and L2 regularization below are
    the primary overfitting controls, with a fixed, conservative
    `num_boost_round`."""
    train_set = lgb.Dataset(np.array(X_train, dtype=float), label=y_train)
    valid_set = lgb.Dataset(np.array(X_valid, dtype=float), label=y_valid, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 100,
        "learning_rate": 0.03,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": 42,
        "deterministic": True,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[valid_set],
    )
    return booster


def raw_predictions(booster: lgb.Booster, X: list[list[float]]) -> list[float]:
    """Raw (uncalibrated) predicted probabilities from the trained model,
    using all trained trees (no early stopping is used, so there is no
    'best iteration' to select)."""
    return list(booster.predict(np.array(X, dtype=float)))


def fit_platt_scaling(raw_probs: list[float], y_true: list[int]) -> tuple[float, float]:
    """Fits a 2-parameter Platt/sigmoid calibration: calibrated_prob =
    sigmoid(a * logit(raw_prob) + b), by minimizing negative log-likelihood
    against y_true. Returns (a, b)."""
    eps = 1e-6
    clipped = [min(max(p, eps), 1 - eps) for p in raw_probs]
    logits = np.array([np.log(p / (1 - p)) for p in clipped])
    y = np.array(y_true, dtype=float)

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        z = a * logits + b
        log_sig = -np.logaddexp(0.0, -z)
        log_one_minus_sig = -np.logaddexp(0.0, z)
        return -np.sum(y * log_sig + (1 - y) * log_one_minus_sig)

    result = minimize(neg_log_likelihood, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    if not result.success:
        raise RuntimeError(f"Platt scaling optimization did not converge: {result.message}")
    a, b = result.x
    return float(a), float(b)


def apply_platt_scaling(raw_probs: list[float], a: float, b: float) -> list[float]:
    """Applies a fitted Platt scaling (a, b) to raw predicted probabilities."""
    eps = 1e-6
    calibrated = []
    for p in raw_probs:
        p_clipped = min(max(p, eps), 1 - eps)
        logit = np.log(p_clipped / (1 - p_clipped))
        z = a * logit + b
        calibrated.append(float(1.0 / (1.0 + np.exp(-z))))
    return calibrated
