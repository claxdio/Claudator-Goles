from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb


def save_model(
    booster: lgb.Booster, platt_params: tuple[float, float], model_dir: Path
) -> None:
    """Persists a trained + calibrated model to `model_dir`: the LightGBM
    booster in its native text format (`booster.txt`) and the Platt
    scaling parameters as `platt.json` (`{"a": ..., "b": ...}`). Creates
    `model_dir` and any missing parent directories."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "booster.txt"))
    a, b = platt_params
    with open(model_dir / "platt.json", "w", encoding="utf-8") as fh:
        json.dump({"a": a, "b": b}, fh)


def load_model(model_dir: Path) -> tuple[lgb.Booster, tuple[float, float]]:
    """Loads a model previously written by `save_model`. Returns
    `(booster, (a, b))`."""
    model_dir = Path(model_dir)
    booster = lgb.Booster(model_file=str(model_dir / "booster.txt"))
    with open(model_dir / "platt.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return booster, (data["a"], data["b"])
