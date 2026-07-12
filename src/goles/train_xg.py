from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from goles.db import get_connection
from goles.xg_model import predict_xg, save_xg_model, train_xg_model

XG_MODEL_PATH = Path("data") / "model" / "xg_booster.txt"


def main() -> None:
    conn = get_connection()
    rows = conn.execute(
        """SELECT xg, is_goal, location_x, location_y, situation, shot_type
           FROM shots WHERE location_x IS NOT NULL"""
    ).fetchall()
    shots = [
        {
            "understat_xg": r[0], "is_goal": bool(r[1]),
            "location_x": r[2], "location_y": r[3],
            "situation": r[4], "shot_type": r[5],
        }
        for r in rows
    ]
    print(f"{len(shots)} tiros historicos cargados.")

    random.seed(42)
    random.shuffle(shots)
    split = int(len(shots) * 0.8)
    train_shots, valid_shots = shots[:split], shots[split:]

    booster = train_xg_model(train_shots)

    valid_np = [s for s in valid_shots if s["situation"] != "Penalty"]
    ours = np.array([predict_xg(booster, s) for s in valid_np])
    theirs = np.array([s["understat_xg"] for s in valid_np])
    actual = np.array([float(s["is_goal"]) for s in valid_np])

    corr = float(np.corrcoef(ours, theirs)[0, 1])
    mae = float(np.mean(np.abs(ours - theirs)))
    print(f"Validacion ({len(valid_np)} tiros no-penal):")
    print(f"  correlacion con xG de Understat: {corr:.4f}  (esperado >= 0.80)")
    print(f"  MAE vs xG de Understat: {mae:.4f}")
    print(f"  media xG nuestro: {ours.mean():.4f} | media xG Understat: {theirs.mean():.4f} | tasa real de gol: {actual.mean():.4f}")

    save_xg_model(booster, XG_MODEL_PATH)
    print(f"Modelo xG guardado en {XG_MODEL_PATH}.")


if __name__ == "__main__":
    main()
