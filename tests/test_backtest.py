import math

from goles.backtest import CUTOFF_MINUTES, BacktestResult, compare_blends, run_backtest
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


def _seed_two_chronological_matches(conn):
    """Two matches for the same home team (Team A) in the same
    league/season, on different dates, so the trailing prior for the
    second match must come from the first match's xG -- never from the
    second match's own shots."""
    records_m1 = [
        {
            "match_id": 501, "league": "TEST", "season": "2025-26", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 20, "team": "home", "xg": 3.0, "is_goal": False,
        },
    ]
    persist_shots(conn, records_m1)

    records_m2 = [
        {
            "match_id": 502, "league": "TEST", "season": "2025-26", "date": "2025-08-10",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 30, "team": "home", "xg": 9.0, "is_goal": True,
        },
        {
            "match_id": 502, "league": "TEST", "season": "2025-26", "date": "2025-08-10",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 78, "team": "away", "xg": 0.4, "is_goal": False,
        },
    ]
    persist_shots(conn, records_m2)
    conn.commit()


def test_run_backtest_uses_trailing_prior_not_same_match_xg():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    result = run_backtest(conn, team="home", cutoff_minutes=[20])
    assert len(result.predicted_probs) == 2

    # Match 501 (Team A's first match this season): trailing prior = 0.0
    # (no earlier match), but its own shot at minute 20 falls inside the
    # (5,20] rolling window, so in_match_xg_recent=3.0 drives the
    # prediction: lambda = 0.5*(3.0/15)*15 = 1.5
    expected_m501 = 1.0 - math.exp(-1.5)

    # Match 502 (Team A's second match): trailing prior = 3.0, taken from
    # match 501 -- NOT match 502's own 9.0 xG shot, which the fix must
    # never use as this match's own prior. No home shots fall inside match
    # 502's (5,20] window (its only home shot is at minute 30), so
    # in_match_xg_recent=0: lambda = 0.5*(3.0/90)*15 = 0.25
    expected_m502 = 1.0 - math.exp(-0.25)

    assert sorted(round(p, 6) for p in result.predicted_probs) == sorted(
        round(p, 6) for p in [expected_m501, expected_m502]
    )


def test_run_backtest_accepts_custom_cutoff_minutes():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    result = run_backtest(conn, team="home", cutoff_minutes=[10, 20, 30])
    assert len(result.predicted_probs) == 2 * 3


def test_compare_blends_returns_one_result_per_blend():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_two_chronological_matches(conn)

    results = compare_blends(conn, team="home", blends=[0.0, 0.5, 1.0], cutoff_minutes=[20])
    assert set(results.keys()) == {0.0, 0.5, 1.0}
    for result in results.values():
        assert len(result.predicted_probs) == 2


def test_load_match_shots_carries_enrichment_fields():
    from goles.backtest import load_match_shots

    conn = get_connection(":memory:")
    init_db(conn)
    persist_shots(conn, [
        {
            "match_id": 601, "league": "TEST", "season": "2526", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 12, "team": "home", "xg": 0.3, "is_goal": False,
            "location_x": 0.9, "location_y": 0.5,
            "situation": "OpenPlay", "shot_type": "Head", "last_action": "Cross",
        },
    ])
    match_id, home_id, away_id = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches"
    ).fetchone()
    shots = load_match_shots(conn, match_id, home_id, away_id)
    assert shots[0]["location_x"] == 0.9
    assert shots[0]["situation"] == "OpenPlay"
    assert shots[0]["shot_type"] == "Head"
    assert shots[0]["last_action"] == "Cross"


def test_load_match_cards_returns_red_card_events():
    from goles.backtest import load_match_cards

    conn = get_connection(":memory:")
    init_db(conn)
    persist_shots(conn, [
        {
            "match_id": 701, "league": "TEST", "season": "2526", "date": "2025-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 10, "team": "home", "xg": 0.1, "is_goal": False,
        },
    ])
    match_id, home_id, away_id = conn.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches"
    ).fetchone()
    conn.execute("INSERT INTO cards (match_id, team_id, minute) VALUES (?, ?, ?)", (match_id, away_id, 55))
    conn.commit()

    cards = load_match_cards(conn, match_id, home_id, away_id)
    assert cards == [{"team": "away", "minute": 55}]
