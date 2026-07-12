from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import numpy as np

# Empirical penalty conversion rate. Penalties are excluded from training
# (a location model trained on open play should not extrapolate to them)
# and assigned this fixed value at prediction time.
PENALTY_XG = 0.76

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32

_SITUATIONS = ["OpenPlay", "FromCorner", "SetPiece", "DirectFreekick"]
_SHOT_TYPES = ["RightFoot", "LeftFoot", "Head", "OtherBodyPart"]

XG_FEATURE_NAMES = (
    ["distance_m", "angle_rad"]
    + [f"situation_{s}" for s in _SITUATIONS]
    + [f"shot_type_{t}" for t in _SHOT_TYPES]
)


def shot_to_features(shot: dict) -> list[float]:
    """Understat-convention shot dict -> xG feature vector. location_x is
    the 0-1 fraction of pitch length toward the attacking goal, location_y
    the 0-1 fraction of pitch width."""
    dx = (1.0 - shot["location_x"]) * PITCH_LENGTH_M
    dy = (shot["location_y"] - 0.5) * PITCH_WIDTH_M
    distance = math.sqrt(dx * dx + dy * dy)

    # Angle subtended by the two goal posts from the shot location.
    half_goal = GOAL_WIDTH_M / 2.0
    denominator = dx * dx + dy * dy - half_goal * half_goal
    if denominator <= 0:
        angle = math.pi / 2  # inside the width of the goal mouth, point blank
    else:
        angle = math.atan2(GOAL_WIDTH_M * dx, denominator)

    features = [distance, angle]
    features += [1.0 if shot.get("situation") == s else 0.0 for s in _SITUATIONS]
    features += [1.0 if shot.get("shot_type") == t else 0.0 for t in _SHOT_TYPES]
    return features


def train_xg_model(shots: list[dict]) -> lgb.Booster:
    """Trains P(goal | shot features) on non-penalty shots. `is_goal` is
    the label; any reference xg on the dicts is never used as an input."""
    usable = [s for s in shots if s.get("situation") != "Penalty"]
    X = np.array([shot_to_features(s) for s in usable], dtype=float)
    y = np.array([int(s["is_goal"]) for s in usable])
    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "learning_rate": 0.05,
        "verbosity": -1,
        "seed": 42,
        "deterministic": True,
    }
    return lgb.train(params, train_set, num_boost_round=300)


def predict_xg(booster: lgb.Booster, shot: dict) -> float:
    if shot.get("situation") == "Penalty":
        return PENALTY_XG
    features = np.array([shot_to_features(shot)], dtype=float)
    return float(booster.predict(features)[0])


def save_xg_model(booster: lgb.Booster, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_xg_model(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
