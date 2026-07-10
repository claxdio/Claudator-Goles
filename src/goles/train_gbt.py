from __future__ import annotations

from goles.backtest import BacktestResult
from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.gbt_model import apply_platt_scaling, fit_platt_scaling, raw_predictions, train_gbt

TEST_SEASON = "2324"
VALIDATION_SEASON = "2223"
# blend=0.1 was the best-performing (least-bad) Poisson configuration found
# in the calibration-improvements plan's real backtest -- used here so the
# Poisson comparison in this report reflects the strongest baseline found
# so far, not the arbitrary original default of 0.5.
POISSON_COMPARISON_BLEND = 0.1


def main() -> None:
    conn = get_connection()
    init_db(conn)

    print("Construyendo el dataset completo desde la base de datos...")
    rows = build_dataset(conn, blend=POISSON_COMPARISON_BLEND)
    print(f"{len(rows)} filas construidas.")

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

    print("\n=== LightGBM (calibrado) en la temporada de test ===")
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
    importances = booster.feature_importance(importance_type="gain")
    for name, importance in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        print(f"  {name}: {importance:.1f}")


if __name__ == "__main__":
    main()
