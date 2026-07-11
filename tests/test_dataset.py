import pytest

from goles.dataset import FEATURE_NAMES, build_dataset, rows_to_arrays, split_by_season
from goles.db import get_connection, init_db
from goles.loaders.understat import persist_shots


def _seed_multi_season_matches(conn):
    """One match per season, three seasons, all featuring Team A at home,
    so build_dataset/split_by_season have something real to work with."""
    records_a = [
        {
            "match_id": 1, "league": "TEST", "season": "SeasonA", "date": "2021-08-01",
            "home_team": "Team A", "away_team": "Team B",
            "minute": 20, "team": "home", "xg": 0.3, "is_goal": False,
        },
    ]
    persist_shots(conn, records_a)

    records_b = [
        {
            "match_id": 2, "league": "TEST", "season": "SeasonB", "date": "2022-08-01",
            "home_team": "Team A", "away_team": "Team C",
            "minute": 25, "team": "home", "xg": 0.5, "is_goal": True,
        },
    ]
    persist_shots(conn, records_b)

    records_c = [
        {
            "match_id": 3, "league": "TEST", "season": "SeasonC", "date": "2023-08-01",
            "home_team": "Team A", "away_team": "Team D",
            "minute": 30, "team": "away", "xg": 0.2, "is_goal": False,
        },
    ]
    persist_shots(conn, records_c)
    conn.commit()


def test_build_dataset_produces_one_row_per_match_team_cutoff():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)

    rows = build_dataset(conn, cutoff_minutes=[20, 25])
    # 3 matches * 2 teams * 2 cutoffs = 12 rows
    assert len(rows) == 12
    assert all(set(FEATURE_NAMES) == set(r.features.keys()) for r in rows)


def test_split_by_season_separates_correctly():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    rows = build_dataset(conn, cutoff_minutes=[20])

    train, validation, test = split_by_season(rows, test_season="SeasonC", validation_season="SeasonB")
    assert all(r.season == "SeasonC" for r in test)
    assert all(r.season == "SeasonB" for r in validation)
    assert all(r.season == "SeasonA" for r in train)
    assert len(train) + len(validation) + len(test) == len(rows)


def test_split_by_season_rejects_same_test_and_validation_season():
    with pytest.raises(ValueError):
        split_by_season([], test_season="X", validation_season="X")


def test_rows_to_arrays_matches_feature_order_and_label():
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    rows = build_dataset(conn, cutoff_minutes=[20])

    X, y = rows_to_arrays(rows)
    assert len(X) == len(rows) == len(y)
    assert len(X[0]) == len(FEATURE_NAMES)
    for row, x_vec in zip(rows, X):
        assert x_vec == [row.features[name] for name in FEATURE_NAMES]
    assert set(y) <= {0, 1}


def test_build_dataset_includes_market_and_rest_features(tmp_path):
    conn = get_connection(":memory:")
    init_db(conn)
    _seed_multi_season_matches(conn)
    # give the SeasonA match's home team (Team A) a market row directly,
    # bypassing the odds loader (out of scope for this test)
    conn.execute(
        """UPDATE matches SET market_home_wp = 0.6, market_draw_wp = 0.25,
           market_away_wp = 0.15, market_over25_wp = 0.55
           WHERE understat_id = 1"""
    )
    conn.commit()

    rows = build_dataset(conn, cutoff_minutes=[20])
    row_a_home = next(r for r in rows if r.match_id == rows[0].match_id and r.team == "home")

    assert "own_rest_days" in row_a_home.features
    assert "opp_rest_days" in row_a_home.features
    assert row_a_home.features["own_market_wp"] == 0.6
    assert row_a_home.features["market_draw_wp"] == 0.25
    assert row_a_home.features["market_over25_wp"] == 0.55
    # Team A's first match of the season -> no prior fixture -> defaulted, not crashed
    assert row_a_home.features["own_rest_days"] == 7.0
    # no cards seeded for this match -> both default to 0
    assert row_a_home.features["own_red_cards"] == 0.0
    assert row_a_home.features["opp_red_cards"] == 0.0
