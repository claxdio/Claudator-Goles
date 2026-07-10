import random
from unittest.mock import patch

import numpy as np
import pytest

from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt


def test_train_gbt_separates_a_simple_deterministic_pattern():
    random.seed(42)
    X_train, y_train = [], []
    for _ in range(400):
        x0 = random.random()
        X_train.append([x0, random.random()])
        y_train.append(1 if x0 > 0.5 else 0)
    X_valid, y_valid = [], []
    for _ in range(100):
        x0 = random.random()
        X_valid.append([x0, random.random()])
        y_valid.append(1 if x0 > 0.5 else 0)

    booster = train_gbt(X_train, y_train, X_valid, y_valid)
    preds = raw_predictions(booster, X_valid)
    assert all(0.0 <= p <= 1.0 for p in preds)
    correct = sum(1 for p, y in zip(preds, y_valid) if (p >= 0.5) == bool(y))
    assert correct / len(y_valid) > 0.85


def test_fit_platt_scaling_recovers_near_identity_for_already_calibrated_probs():
    random.seed(0)
    raw_probs = []
    y_true = []
    for _ in range(500):
        p = random.random()
        raw_probs.append(p)
        y_true.append(1 if random.random() < p else 0)

    a, b = fit_platt_scaling(raw_probs, y_true)
    calibrated = apply_platt_scaling(raw_probs, a, b)
    mean_abs_diff = sum(abs(c - p) for c, p in zip(calibrated, raw_probs)) / len(raw_probs)
    assert mean_abs_diff < 0.1


def test_platt_scaling_corrects_a_systematically_overconfident_model():
    random.seed(1)
    true_rate = 0.3
    n = 500
    y_true = [1 if random.random() < true_rate else 0 for _ in range(n)]
    raw_probs = [0.8] * n  # a badly overconfident constant prediction

    a, b = fit_platt_scaling(raw_probs, y_true)
    calibrated = apply_platt_scaling(raw_probs, a, b)
    assert abs(calibrated[0] - true_rate) < abs(0.8 - true_rate)


def test_fit_platt_scaling_raises_when_optimizer_does_not_converge():
    class FakeResult:
        success = False
        message = "mock optimizer failure"
        x = np.array([1.0, 0.0])

    with patch("goles.gbt_model.minimize", return_value=FakeResult()):
        with pytest.raises(RuntimeError, match="mock optimizer failure"):
            fit_platt_scaling([0.1, 0.9], [0, 1])
