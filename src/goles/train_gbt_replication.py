from __future__ import annotations

from goles.backtest import BacktestResult
from goles.dataset import build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt

# Repeatability check: does the same pipeline show a real edge on a
# DIFFERENT held-out season, not just 2324? To keep this a fair
# walk-forward simulation (train on the past, evaluate on the future --
# the same discipline a live system would actually follow), 2324 is
# dropped entirely from this run, not just excluded from training: keeping
# it in the training pool would mean training on data chronologically
# AFTER the test season, which is not how a real deployment would ever
# look.
TEST_SEASON = "2223"
VALIDATION_SEASON = "2122"
ELIGIBLE_SEASONS = {"1819", "1920", "2021", "2122", "2223"}
POISSON_COMPARISON_BLEND = 0.1


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print("Construyendo el dataset completo desde la base de datos...")
    rows = build_dataset(conn, blend=POISSON_COMPARISON_BLEND)
    rows = [r for r in rows if r.season in ELIGIBLE_SEASONS]
    print(f"{len(rows)} filas construidas (excluyendo 2324 por completo, incluso del entrenamiento).")

    train_rows, valid_rows, test_rows = split_by_season(rows, TEST_SEASON, VALIDATION_SEASON)
    print(f"Train: {len(train_rows)}  Validation: {len(valid_rows)}  Test: {len(test_rows)}")

    X_train, y_train = rows_to_arrays(train_rows)
    X_valid, y_valid = rows_to_arrays(valid_rows)
    X_test, y_test = rows_to_arrays(test_rows)

    print("Entrenando LightGBM...")
    booster = train_gbt(X_train, y_train, X_valid, y_valid)

    print("Calibrando con Platt scaling sobre el set de validacion...")
    valid_raw = raw_predictions(booster, X_valid)
    a, b = fit_platt_scaling(valid_raw, y_valid)

    print("Evaluando en la temporada de test (nunca vista durante entrenamiento ni calibracion)...")
    test_raw = raw_predictions(booster, X_test)
    test_calibrated = apply_platt_scaling(test_raw, a, b)

    gbt_result = BacktestResult(
        predicted_probs=test_calibrated,
        actual_outcomes=[bool(y) for y in y_test],
    )

    print(f"\n=== LightGBM (calibrado) en la temporada de test {TEST_SEASON} (replica) ===")
    print(f"Muestras evaluadas: {len(gbt_result.predicted_probs)}")
    print(f"Brier score: {gbt_result.brier_score:.4f}")
    print(f"Brier score (base ingenua): {gbt_result.no_skill_brier_score:.4f}")
    print(f"Brier Skill Score: {gbt_result.brier_skill_score:.4f}  (>0 = mejor que la base ingenua)")
    print("Calibracion (bin_low, prob. media predicha, frecuencia real, n):")
    for bin_low, mean_pred, mean_actual, count in gbt_result.calibration_bins():
        print(f"  [{bin_low:.1f}-{bin_low + 0.2:.1f}) pred={mean_pred:.3f} real={mean_actual:.3f} n={count}")

    poisson_test_probs = [r.features["poisson_prob"] for r in test_rows]
    poisson_result = BacktestResult(
        predicted_probs=poisson_test_probs,
        actual_outcomes=[bool(y) for y in y_test],
    )
    print(f"\n=== Poisson baseline (blend={POISSON_COMPARISON_BLEND}), misma temporada de test ===")
    print(f"Brier Skill Score (Poisson): {poisson_result.brier_skill_score:.4f}")

    print("\n=== Importancia de features (LightGBM, ganancia total) ===")
    from goles.dataset import FEATURE_NAMES

    importances = booster.feature_importance(importance_type="gain")
    for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.1f}")


if __name__ == "__main__":
    main()
