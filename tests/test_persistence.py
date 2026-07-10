import random

from goles.gbt_model import fit_platt_scaling, raw_predictions, train_gbt
from goles.persistence import load_model, save_model


def _train_tiny_booster():
    random.seed(7)
    X_train, y_train = [], []
    for _ in range(200):
        x0 = random.random()
        X_train.append([x0, random.random()])
        y_train.append(1 if x0 > 0.5 else 0)
    X_valid, y_valid = [], []
    for _ in range(50):
        x0 = random.random()
        X_valid.append([x0, random.random()])
        y_valid.append(1 if x0 > 0.5 else 0)
    booster = train_gbt(X_train, y_train, X_valid, y_valid)
    return booster, X_valid, y_valid


def test_save_and_load_model_round_trips_predictions_and_platt_params(tmp_path):
    booster, X_valid, y_valid = _train_tiny_booster()
    raw_valid = raw_predictions(booster, X_valid)
    a, b = fit_platt_scaling(raw_valid, y_valid)

    model_dir = tmp_path / "model"
    save_model(booster, (a, b), model_dir)
    loaded_booster, (loaded_a, loaded_b) = load_model(model_dir)

    assert abs(loaded_a - a) < 1e-9
    assert abs(loaded_b - b) < 1e-9
    original_preds = raw_predictions(booster, X_valid)
    loaded_preds = raw_predictions(loaded_booster, X_valid)
    assert len(original_preds) == len(loaded_preds)
    for p_orig, p_loaded in zip(original_preds, loaded_preds):
        assert abs(p_orig - p_loaded) < 1e-6


def test_save_model_creates_missing_parent_directories(tmp_path):
    booster, X_valid, y_valid = _train_tiny_booster()
    a, b = fit_platt_scaling(raw_predictions(booster, X_valid), y_valid)

    nested_dir = tmp_path / "nested" / "model"
    save_model(booster, (a, b), nested_dir)
    assert (nested_dir / "booster.txt").exists()
    assert (nested_dir / "platt.json").exists()
