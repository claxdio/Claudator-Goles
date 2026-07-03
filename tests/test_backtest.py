import math

from goles.backtest import CUTOFF_MINUTES, BacktestResult, run_backtest
from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots


def _seed_one_match(conn):
    records = []
    for m, xg, goal in [(10, 0.1, False), (30, 0.4, True), (55, 0.3, False), (78, 0.5, True)]:
        records.append(
            {
                "match_id": 1, "league": "TEST", "season": "2025-26",
                "home_team": "Team A", "away_team": "Team B",
                "minute": m, "team": "home", "xg": xg, "is_goal": goal,
            }
        )
    for m, xg, goal in [(20, 0.2, False), (65, 0.35, False)]:
        records.append(
            {
                "match_id": 1, "league": "TEST", "season": "2025-26",
                "home_team": "Team A", "away_team": "Team B",
                "minute": m, "team": "away", "xg": xg, "is_goal": goal,
            }
        )
    persist_shots(conn, records)


def test_run_backtest_returns_one_prediction_per_cutoff():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_one_match(conn)
    result = run_backtest(conn, team="home")
    assert len(result.predicted_probs) == len(CUTOFF_MINUTES)
    assert len(result.actual_outcomes) == len(CUTOFF_MINUTES)
    assert all(0.0 <= p <= 1.0 for p in result.predicted_probs)


def test_brier_score_is_between_zero_and_one():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_one_match(conn)
    result = run_backtest(conn, team="home")
    assert 0.0 <= result.brier_score <= 1.0


def test_backtest_result_handles_empty_data():
    empty = BacktestResult(predicted_probs=[], actual_outcomes=[])
    assert math.isnan(empty.brier_score)
    assert empty.calibration_bins() == []


def test_calibration_bins_groups_by_predicted_probability():
    result = BacktestResult(
        predicted_probs=[0.05, 0.12, 0.55, 0.58],
        actual_outcomes=[False, True, True, False],
    )
    bins = result.calibration_bins(n_bins=5)
    # two predictions land in bin [0.0, 0.2), two land in bin [0.4, 0.6)
    bin_counts = {round(b[0], 1): b[3] for b in bins}
    assert bin_counts[0.0] == 2
    assert bin_counts[0.4] == 2


def test_no_skill_brier_score_matches_base_rate_formula():
    result = BacktestResult(
        predicted_probs=[0.1, 0.9, 0.1, 0.9],
        actual_outcomes=[False, True, True, False],
    )
    # base rate = 2/4 = 0.5 -> no-skill brier = 0.5 * 0.5 = 0.25
    assert abs(result.no_skill_brier_score - 0.25) < 1e-9


def test_no_skill_brier_score_handles_empty_data():
    result = BacktestResult(predicted_probs=[], actual_outcomes=[])
    assert math.isnan(result.no_skill_brier_score)


def test_brier_skill_score_is_one_for_perfect_predictions():
    result = BacktestResult(
        predicted_probs=[0.0, 1.0, 0.0, 1.0],
        actual_outcomes=[False, True, False, True],
    )
    assert abs(result.brier_skill_score - 1.0) < 1e-9


def test_brier_skill_score_is_zero_when_model_matches_naive_baseline():
    # always predicting exactly the base rate (0.5) makes model_brier == no_skill_brier
    result = BacktestResult(
        predicted_probs=[0.5, 0.5, 0.5, 0.5],
        actual_outcomes=[False, True, True, False],
    )
    assert abs(result.brier_skill_score - 0.0) < 1e-9
